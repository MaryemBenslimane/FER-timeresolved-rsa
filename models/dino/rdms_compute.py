import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# -----------------------------
# Thread safety on clusters
# -----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import numpy as np
import h5py
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import autocast

try:
    import cv2
except ImportError:
    cv2 = None


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _to_numpy(output: torch.Tensor) -> np.ndarray:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor activation, got {type(output)}")
    return output.detach().cpu().to(torch.float32).numpy()


# =========================================================
# Video loading
# =========================================================

def sample_uniform_frame_indices(num_total_frames: int, num_samples: int) -> np.ndarray:
    if num_total_frames <= 0:
        raise ValueError("num_total_frames must be > 0")
    num_samples = min(num_samples, num_total_frames)
    return np.linspace(0, num_total_frames - 1, num=num_samples, dtype=int)


def load_video_frames(video_path: str, num_frames: int = 16, to_rgb: bool = True) -> List[Image.Image]:
    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        target_indices = set(sample_uniform_frame_indices(total_frames, num_frames).tolist())

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in target_indices:
                if to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))

            frame_idx += 1

        cap.release()
    else:
        from torchvision.io import read_video
        video, _, _ = read_video(video_path, pts_unit="sec")
        total_frames = int(video.shape[0])
        if total_frames <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        target_indices = sample_uniform_frame_indices(total_frames, num_frames)
        frames = []
        for idx in target_indices:
            frame = video[int(idx)].numpy()
            if not to_rgb:
                frame = frame[..., ::-1]
            frames.append(Image.fromarray(frame))

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    return frames


# =========================================================
# Stimulus dataset
# =========================================================

class StimulusVideoDataset(Dataset):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        video_paths: Optional[List[str]] = None,
        num_frames: int = 16,
        image_size: int = 224,
    ):
        self.num_frames = num_frames

        if video_paths is not None:
            self.samples = [str(Path(p)) for p in video_paths]
        else:
            if root_dir is None:
                raise ValueError("Provide either root_dir or video_paths.")
            root = Path(root_dir)
            self.samples = sorted(
                [str(p) for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
            )

        if len(self.samples) == 0:
            raise ValueError("No stimulus videos found.")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path = self.samples[idx]
        frames = load_video_frames(video_path, num_frames=self.num_frames)
        frame_tensors = [self.transform(frame) for frame in frames]
        video = torch.stack(frame_tensors, dim=0)  # [T, C, H, W]

        return {
            "video": video,
            "path": video_path,
            "index": idx,
        }


def collate_stimulus_videos(batch):
    videos = torch.stack([x["video"] for x in batch], dim=0)
    paths = [x["path"] for x in batch]
    indices = torch.tensor([x["index"] for x in batch], dtype=torch.long)
    return {"video": videos, "path": paths, "index": indices}


# =========================================================
# Model
# =========================================================

class LearnedTemporalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pos_embed[:, :T, :]


class DinoV2TemporalTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int = 16,
        dino_model_name: str = "dinov2_vitb14",
        proj_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        freeze_dino: bool = True,
    ):
        super().__init__()

        self.dino = torch.hub.load("facebookresearch/dinov2", dino_model_name)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            dino_out = self.dino(dummy)
            dino_dim = dino_out.shape[-1]

        if freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad = False

        self.num_frames = num_frames
        self.dino_dim = dino_dim
        self.proj_dim = proj_dim

        self.frame_proj = nn.Linear(dino_dim, proj_dim)
        self.pos_enc = LearnedTemporalPositionalEncoding(max_len=num_frames + 1, dim=proj_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=int(proj_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(proj_dim)

        self.frame_classifier = nn.Linear(proj_dim, num_classes)
        self.video_classifier = nn.Linear(proj_dim, num_classes)

    def extract_frame_features(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        flat = video.view(B * T, C, H, W)
        feats = self.dino(flat)
        feats = feats.view(B, T, -1)
        return feats

    def forward_features(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_frame_features = self.extract_frame_features(video)
        projected = self.frame_proj(raw_frame_features)

        B, T, D = projected.shape
        cls = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls, projected], dim=1)
        x = self.pos_enc(x)
        x = self.temporal_transformer(x)
        x = self.norm(x)

        video_embedding = x[:, 0, :]
        frame_embeddings = x[:, 1:, :]

        return {
            "raw_frame_features": raw_frame_features,
            "projected_frame_features": projected,
            "transformer_tokens": x,
            "frame_embeddings": frame_embeddings,
            "video_embedding": video_embedding,
        }

    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.forward_features(video)
        feats["frame_logits"] = self.frame_classifier(feats["frame_embeddings"])
        feats["video_logits"] = self.video_classifier(feats["video_embedding"])
        return feats


# =========================================================
# Layer activation collector with hooks
# =========================================================

class LayerActivationCollector:
    def __init__(self, modules: Dict[str, nn.Module]):
        self.modules = modules
        self.current_batch: Dict[str, np.ndarray] = {}
        self.handles = [
            module.register_forward_hook(self._make_hook(name))
            for name, module in modules.items()
        ]

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self.current_batch[name] = _to_numpy(output)
        return hook

    def clear(self):
        self.current_batch = {}

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def build_layer_module_map(model: DinoV2TemporalTransformer) -> Dict[str, nn.Module]:
    modules: Dict[str, nn.Module] = {}

    for idx, block in enumerate(model.dino.blocks):
        modules[f"dino_blocks_{idx}"] = block

    modules["dino_norm"] = model.dino.norm
    modules["frame_proj"] = model.frame_proj
    modules["pos_enc"] = model.pos_enc

    for idx, layer in enumerate(model.temporal_transformer.layers):
        modules[f"temporal_transformer_layers_{idx}"] = layer

    modules["temporal_norm"] = model.norm
    modules["frame_classifier"] = model.frame_classifier
    modules["video_classifier"] = model.video_classifier

    return modules


def reshape_layer_activation(
    layer_name: str,
    activation: np.ndarray,
    batch_size: int,
    num_frames: int,
) -> np.ndarray:
    if activation.shape[0] == batch_size * num_frames:
        return activation.reshape(batch_size, num_frames, *activation.shape[1:])
    if activation.shape[0] == batch_size:
        return activation
    raise ValueError(
        f"Unexpected leading dimension for {layer_name}: "
        f"{activation.shape[0]} (batch_size={batch_size}, num_frames={num_frames})"
    )


# =========================================================
# RDM computation
# =========================================================

def correlation_rdm(features: np.ndarray) -> np.ndarray:
    """
    features: [N, D]
    returns: [N, N] correlation distance = 1 - Pearson correlation
    """
    corr = np.corrcoef(features)
    rdm = 1.0 - corr
    return np.nan_to_num(rdm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def flatten_activation_to_2d(layer_name: str, activation: np.ndarray) -> np.ndarray:
    """
    Collapse a per-stimulus activation to a single vector per stimulus.
    Input: [N, ...] -> Output: [N, D_flat]

    Strategy per activation shape:
    - [N, D]              -> use as-is
    - [N, T, D] (temporal)-> concatenate CLS + mean-of-frame-tokens
    - [N, T, D] (other)   -> mean over T
    - [N, T, P, D]        -> mean over all patches, then mean over frames
                             (uses ALL tokens, not just CLS)
    """
    N = activation.shape[0]
    if activation.ndim == 2:
        return activation
    elif activation.ndim == 3:
        if "temporal_transformer" in layer_name or layer_name in {"temporal_norm"}:
            # CLS at pos 0, frame tokens at pos 1:
            # Concatenate CLS + mean-of-frames for richer representation
            cls = activation[:, 0, :]             # [N, D]
            frame_mean = activation[:, 1:, :].mean(axis=1)  # [N, D]
            return np.concatenate([cls, frame_mean], axis=1)  # [N, 2D]
        elif layer_name == "pos_enc":
            # Positional encoding adds the same thing to all stimuli,
            # but the input differs per stimulus. Use mean over all tokens.
            return activation.mean(axis=1)
        else:
            # frame_proj, frame_classifier: mean over frames
            return activation.mean(axis=1)
    elif activation.ndim == 4:
        # DINO blocks: [N, T, P, D] where P = 257 (1 CLS + 256 patches)
        # Use mean over ALL tokens (CLS + patches), then mean over frames
        # This preserves spatial information from all patches
        return activation.mean(axis=(1, 2))  # [N, D]
    else:
        return activation.reshape(N, -1)


# =========================================================
# Main extraction + RDM pipeline
# =========================================================

@torch.no_grad()
def extract_and_compute_rdms(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    layer_collector: LayerActivationCollector,
    idx_to_class: Optional[Dict[int, str]] = None,
) -> Tuple[Dict[str, np.ndarray], List[dict], List[str]]:
    """
    Run model on all stimuli, collect per-layer activations,
    then compute one RDM (18x18) per layer.

    Returns:
        layer_activations: {layer_name: [N, ...]} full activations
        predictions: list of prediction dicts
        paths: ordered stimulus paths
    """
    model.eval()

    # Accumulate activations per layer across batches
    layer_accum: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    all_predictions = []
    all_paths = []
    all_indices = []

    for batch in loader:
        videos = batch["video"].to(device)
        paths = batch["path"]
        indices = batch["index"].cpu().numpy()
        batch_size, num_frames = videos.shape[:2]

        layer_collector.clear()

        with autocast(device_type="cuda", dtype=torch.float16):
            out = model(videos)

        # Collect predictions
        frame_logits = out["frame_logits"].float()
        video_logits = out["video_logits"].float()
        frame_probs = torch.softmax(frame_logits, dim=-1)
        video_probs = torch.softmax(video_logits, dim=-1)
        frame_preds = frame_probs.argmax(dim=-1).cpu().numpy()
        video_preds = video_probs.argmax(dim=-1).cpu().numpy()

        for i in range(len(paths)):
            vp = int(video_preds[i])
            fp = frame_preds[i].tolist()
            pred_item = {
                "stimulus_index": int(indices[i]),
                "video": paths[i],
                "video_prediction_index": vp,
                "video_prediction_name": idx_to_class[vp] if idx_to_class else str(vp),
                "video_probabilities": video_probs[i].cpu().tolist(),
                "frame_predictions_index": fp,
                "frame_predictions_name": [
                    idx_to_class[int(x)] if idx_to_class else str(int(x))
                    for x in fp
                ],
                "frame_probabilities": frame_probs[i].cpu().tolist(),
            }
            all_predictions.append(pred_item)

        # Collect layer activations
        for layer_name, activation in layer_collector.current_batch.items():
            reshaped = reshape_layer_activation(layer_name, activation, batch_size, num_frames)
            if layer_name not in layer_accum:
                layer_accum[layer_name] = []
            layer_accum[layer_name].append((indices, reshaped))

        all_paths.extend(paths)
        all_indices.append(indices)

    # Reorder everything by stimulus index
    all_indices = np.concatenate(all_indices)
    order = np.argsort(all_indices)
    ordered_paths = [all_paths[i] for i in order]

    pred_map = {item["stimulus_index"]: item for item in all_predictions}
    ordered_predictions = [pred_map[int(i)] for i in sorted(pred_map.keys())]

    # Reassemble layer activations in stimulus order
    num_stimuli = len(ordered_paths)
    layer_activations: Dict[str, np.ndarray] = {}
    for layer_name, chunks in layer_accum.items():
        # Figure out shape from first chunk
        first_act = chunks[0][1]
        full_shape = (num_stimuli,) + first_act.shape[1:]
        full = np.zeros(full_shape, dtype=np.float32)
        for idxs, act in chunks:
            for local_i, global_i in enumerate(idxs):
                full[global_i] = act[local_i]
        layer_activations[layer_name] = full

    return layer_activations, ordered_predictions, ordered_paths


def main():
    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------
    checkpoint_path = "/home/maryem/scratch/FER/DINOv2/best_model.pt"
    stimulus_dir = "/home/maryem/scratch/FER/Stims_Videos/Stims_Videos"
    out_dir = Path("/home/maryem/scratch/FER/DINOv2")
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 4
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config", {})
    class_to_idx = ckpt.get("class_to_idx", None)

    if class_to_idx is not None:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
    else:
        raise ValueError("Checkpoint must contain class_to_idx.")

    num_frames = config.get("num_frames", 16)
    image_size = config.get("image_size", 224)
    proj_dim = config.get("proj_dim", 256)
    num_classes = config.get("num_classes", num_classes)

    # -----------------------------------------------------
    # Rebuild model
    # -----------------------------------------------------
    model = DinoV2TemporalTransformer(
        num_classes=num_classes,
        num_frames=num_frames,
        dino_model_name="dinov2_vitb14",
        proj_dim=proj_dim,
        num_layers=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.2,
        freeze_dino=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # -----------------------------------------------------
    # Stimulus dataset/loader
    # -----------------------------------------------------
    dataset = StimulusVideoDataset(
        root_dir=stimulus_dir,
        num_frames=num_frames,
        image_size=image_size,
    )

    print(f"Found {len(dataset)} stimulus videos")
    for i, s in enumerate(dataset.samples):
        print(f"  [{i:2d}] {Path(s).name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_stimulus_videos,
        persistent_workers=True,
    )

    # -----------------------------------------------------
    # Register hooks on all layers
    # -----------------------------------------------------
    layer_module_map = build_layer_module_map(model)
    layer_collector = LayerActivationCollector(layer_module_map)

    print(f"\nRegistered hooks on {len(layer_module_map)} layers:")
    for name in layer_module_map:
        print(f"  {name}")

    # -----------------------------------------------------
    # Extract activations and predictions
    # -----------------------------------------------------
    print("\nExtracting activations...")
    layer_activations, predictions, paths = extract_and_compute_rdms(
        model=model,
        loader=loader,
        device=device,
        layer_collector=layer_collector,
        idx_to_class=idx_to_class,
    )
    layer_collector.remove()

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------
    with open(out_dir / "stimulus_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nSaved predictions for {len(predictions)} stimuli")

    # Save paths
    with open(out_dir / "stimulus_paths.txt", "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")

    # -----------------------------------------------------
    # Save individual layer activations as .npy
    # -----------------------------------------------------
    act_dir = out_dir / "layer_activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    print("\nLayer activation shapes:")
    for layer_name in sorted(layer_activations.keys()):
        act = layer_activations[layer_name]
        np.save(act_dir / f"{layer_name}.npy", act)
        print(f"  {layer_name}: {act.shape}")

    # Also save summary
    summary = {name: list(act.shape) for name, act in layer_activations.items()}
    with open(act_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # -----------------------------------------------------
    # Compute RDMs: one (18, 18) per layer -> stack to (18, 18, n_layers)
    # -----------------------------------------------------
    print("\nComputing layer RDMs (18 x 18 x n_layers)...")

    # Define consistent layer order
    layer_order = sorted(layer_activations.keys())
    n_stimuli = len(paths)
    n_layers = len(layer_order)

    rdm_stack = np.zeros((n_stimuli, n_stimuli, n_layers), dtype=np.float32)

    for li, layer_name in enumerate(layer_order):
        act = layer_activations[layer_name]
        # Flatten each stimulus activation to a single vector
        flat = flatten_activation_to_2d(layer_name, act)
        rdm = correlation_rdm(flat)
        rdm_stack[:, :, li] = rdm
        print(f"  [{li:2d}] {layer_name}: activation {act.shape} -> flat {flat.shape} -> RDM {rdm.shape}")

    # Save stacked RDMs
    np.save(out_dir / "layer_rdms_stacked.npy", rdm_stack)

    # Save metadata
    rdm_meta = {
        "shape": list(rdm_stack.shape),
        "description": f"({n_stimuli}, {n_stimuli}, {n_layers}) - one correlation-distance RDM per layer",
        "layer_order": layer_order,
        "layer_index": {name: i for i, name in enumerate(layer_order)},
        "metric": "correlation_distance",
        "stimulus_paths": paths,
    }
    with open(out_dir / "layer_rdms_stacked.json", "w", encoding="utf-8") as f:
        json.dump(rdm_meta, f, indent=2)

    print(f"\nFinal stacked RDMs shape: {rdm_stack.shape}")
    print(f"Saved to: {out_dir / 'layer_rdms_stacked.npy'}")
    print(f"Metadata: {out_dir / 'layer_rdms_stacked.json'}")
    print(f"Layer order: {layer_order}")
    print("\nDone.")


if __name__ == "__main__":
    main()
