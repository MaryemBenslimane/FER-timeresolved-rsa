"""RDMs for the facenet-pytorch FaceNet ("Complete VGGFace2") and its AffectNet fine-tune.

Kept separate from extract_model_rdms.py because this model is facenet-pytorch's
InceptionResnetV1 with last_linear/last_bn removed (1792 -> 3 head), which that script
does not build, and because it uses the Stimuliloader normalization. Three weight sets,
one architecture, so the RDMs share layer coordinates:

  default        public VGGFace2 backbone + random 3-class head   ("Complete VGGFace2")
  --finetuned    an AffectNet fine-tuned checkpoint               ("AffectNet FT")
  --random       all weights Xavier-randomized (seed 42)

`--model vggface` is FaceNet (not VGG16). `--model resnet18` (ImageNet ResNet18) is a
legacy path; the paper's ResNet18 rows come from extract_model_rdms.py.
Output: <out-name>.npy (n_layer, n_slides, 18, 18) + .predictions.csv + .layers.json.

Usage (from models/, as in analysis/layer_matched/run_layer_matched.sh):
  python image/extract_image_models_16frames.py --model vggface --layers weights \
      --hdf5-dir ../stim_frames --n-slides 16 --out-name vggface_affectnet_16f_pretrained_weights
  python image/extract_image_models_16frames.py --model vggface --layers weights \
      --finetuned ../net_weights/best_inceptionresnetv1_fer.pt \
      --hdf5-dir ../stim_frames --n-slides 16 --out-name vggface_affectnet_16f_finetuned_weights
"""
import os, sys, argparse, csv, json
from collections import defaultdict
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules as mod
from torchvision import models

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.load_data_FER import Stimuliloader
from facenet_pytorch import InceptionResnetV1
from FaceNet import FaceNet, Block35, Mixed_6a, Mixed_7a, Block8, Block17

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.environ.get("FER_ROOT", ".")
HDF5 = os.path.join(ROOT, "stim_frames")
OUT_DIR = os.path.join(ROOT, "analysis/model_rdms_canonical")

# Identical to models/extract_model_activations.py
LAYER_TYPES = [mod.activation.PReLU, mod.batchnorm.BatchNorm1d, mod.conv.Conv2d,
               mod.linear.Linear, mod.batchnorm.BatchNorm2d, mod.pooling.AdaptiveAvgPool2d,
               mod.activation.ReLU, mod.pooling.MaxPool2d, mod.activation.ReLU6,
               Block35, Mixed_6a, Mixed_7a, Block8, Block17]


class InceptionResnetV1_Custom(InceptionResnetV1):
    """EXACT architecture of the trained VGGFace model (built via pretrained=None,
    378 hookable layers). Backbone weights are copied in separately."""
    def __init__(self, num_classes=3):
        super().__init__(pretrained=None, classify=True, num_classes=num_classes)
        self.last_linear = nn.Identity()
        self.last_bn = nn.Identity()
        self.logits = nn.Linear(1792, num_classes)


def randomize_all_weights(model, seed=42):
    """Re-initialize weights module-by-module (Xavier-uniform for conv/linear,
    identity for normalization layers) matching the fully-random control. Zeroing
    BatchNorm gamma would collapse activations to constants and yield degenerate
    (NaN) RDMs, so normalization layers are set to the identity transform."""
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
            if hasattr(m, "running_mean") and m.running_mean is not None:
                m.running_mean.zero_(); m.running_var.fill_(1.0)
    return model


def build_model(name, finetuned=None, random=False):
    """finetuned=<ckpt path> -> load the AffectNet-fine-tuned weights (FINE-TUNED);
    random=True -> identical architecture, all weights Xavier-randomized (RANDOM control);
    otherwise -> pre-fine-tuning backbone (ImageNet / VGGFace2) + random head.
    All three use the identical architecture so the RDMs share coordinates."""
    if random:
        torch.manual_seed(42)
        if name == "vggface":
            m = InceptionResnetV1_Custom(num_classes=3)
        elif name == "resnet18":
            m = models.resnet18(weights=None)
            m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, 3))
        else:
            raise ValueError(name)
        randomize_all_weights(m, seed=42)
        print(f"[{name} RANDOM] all weights Xavier-randomized (seed 42)")
        return m.to(DEVICE).eval()
    if name == "vggface":
        m = InceptionResnetV1_Custom(num_classes=3)   # arch identical in both modes
        if finetuned:
            state = torch.load(finetuned, map_location="cpu")
            if any(k.startswith("module.") for k in state):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            miss, unexp = m.load_state_dict(state, strict=False)
            print(f"[vggface FT] loaded {finetuned} (missing {len(miss)}, unexpected {len(unexp)})")
        else:
            vgg2 = InceptionResnetV1(pretrained='vggface2')
            sd, src = m.state_dict(), vgg2.state_dict()
            copied = sum(1 for k in sd if k in src and src[k].shape == sd[k].shape)
            for k in sd:
                if k in src and src[k].shape == sd[k].shape: sd[k] = src[k]
            m.load_state_dict(sd)
            print(f"[vggface PRE] copied {copied}/{len(sd)} backbone tensors from VGGFace2")
    elif name == "resnet18":
        if finetuned:
            m = models.resnet18(weights=None)
            m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, 3))
            state = torch.load(finetuned, map_location="cpu")
            if any(k.startswith("module.") for k in state):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            miss, unexp = m.load_state_dict(state, strict=False)
            print(f"[resnet18 FT] loaded {finetuned} (missing {len(miss)}, unexpected {len(unexp)})")
        else:
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, 3))  # random head
    else:
        raise ValueError(name)
    return m.to(DEVICE).eval()


def corr_distance_rdm(X):
    """X: (18, feat) -> (18,18) correlation distance 1 - Pearson r."""
    Xc = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    denom = np.maximum(n * n.T, 1e-12)
    cor = (Xc @ Xc.T) / denom
    rdm = 1.0 - cor
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def extract(name, finetuned=None, random=False, hdf5_dir=None, n_slides=15,
            stim_names=None, batch_size=6, layers="all"):
    hdf5_dir = hdf5_dir or HDF5
    model = build_model(name, finetuned=finetuned, random=random)
    acts = {}
    handles = []
    selected_types = (nn.Conv2d, nn.Linear) if layers == "weights" else tuple(LAYER_TYPES)
    for lname, layer in model.named_modules():
        if isinstance(layer, selected_types):
            def mk(k):
                def hook(_m, _i, o):
                    acts[k] = o.detach().cpu().numpy().reshape(o.size(0), -1)
                return hook
            handles.append(layer.register_forward_hook(mk(lname)))

    per_slide, predictions = [], []
    layer_order = None
    for s in range(1, n_slides + 1):
        with h5py.File(f"{hdf5_dir}/Slide{s}.h5", "r") as f:
            n_stim = int(f["images"].shape[0])
        loader = Stimuliloader(batch_size, f"{hdf5_dir}/Slide{s}.h5")
        chunks, pred_chunks = defaultdict(list), []
        for imgs in loader:
            imgs = imgs.to(DEVICE)
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            acts.clear()
            with torch.no_grad():
                output = model(imgs)
            if layer_order is None:
                layer_order = list(acts.keys())
            for key in layer_order:
                chunks[key].append(acts[key])
            pred_chunks.append(output.argmax(1).cpu().numpy())
        slide_acts = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
        rdms = np.stack([corr_distance_rdm(slide_acts[k]) for k in layer_order], axis=0)
        per_slide.append(rdms)
        pred = np.concatenate(pred_chunks)
        names = stim_names or [f"stim_{i:02d}" for i in range(n_stim)]
        predictions.extend((names[i], s, int(pred[i])) for i in range(n_stim))
        print(f"  slide {s}: {rdms.shape[0]} layers", flush=True)

    for h in handles:
        h.remove()
    stacked = np.stack(per_slide, axis=1)  # (L, 15, 18, 18)
    return stacked, predictions, layer_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "vggface"])
    ap.add_argument("--finetuned", default=None, help="path to fine-tuned checkpoint; omit for pretrained baseline")
    ap.add_argument("--random", action="store_true", help="fully Xavier-randomized weights (matched-coord random control)")
    ap.add_argument("--hdf5-dir", default=None, help="stimulus slide directory (default: legacy 15-slide hdf5/)")
    ap.add_argument("--n-slides", type=int, default=15, help="number of Slide{i}.h5 files = model-time points")
    ap.add_argument("--out-name", default=None, help="output basename (without .npy); overrides the default naming")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--layers", choices=["all", "weights"], default="all")
    args = ap.parse_args()
    tag = {"resnet18": "resnet18_affectnet", "vggface": "vggface_affectnet"}[args.model]
    suffix = "random_matched" if args.random else ("finetuned_matched" if args.finetuned else "pretrained")
    base = args.out_name if args.out_name else f"{tag}_{suffix}"
    manifest_path = args.manifest or os.path.join(args.hdf5_dir or HDF5, "stimulus_manifest.json")
    manifest = json.loads(open(manifest_path).read()) if os.path.exists(manifest_path) else None
    stim_names = [row["stimulus"] for row in manifest] if manifest else None
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{base}.npy")
    rdm, predictions, layer_order = extract(
        args.model, finetuned=args.finetuned, random=args.random,
        hdf5_dir=args.hdf5_dir, n_slides=args.n_slides, stim_names=stim_names,
        batch_size=args.batch_size, layers=args.layers)
    print(f"[{args.model}] {suffix} RDM shape = {rdm.shape}")
    np.save(out, rdm)
    with open(os.path.join(args.out_dir, f"{base}.predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stim", "frame", "pred_idx"]); w.writerows(predictions)
    with open(os.path.join(args.out_dir, f"{base}.layers.json"), "w") as f:
        json.dump({"shape": list(rdm.shape), "layer_order": layer_order,
                   "model": args.model, "finetuned": args.finetuned,
                   "random": args.random, "stimuli": stim_names,
                   "stimulus_manifest": manifest_path, "layers": args.layers}, f, indent=2)
    print(f"[ok] saved -> {out}")


if __name__ == "__main__":
    main()
