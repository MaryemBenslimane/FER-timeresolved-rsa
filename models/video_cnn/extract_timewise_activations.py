import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from models import ResNet2DTemporal, Small3DCNN

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def build_model(model_name: str, temporal: str = "mean") -> torch.nn.Module:
    if model_name == "cnn3d":
        return Small3DCNN(num_classes=3)
    return ResNet2DTemporal(num_classes=3, temporal=temporal)


def load_model(
    checkpoint_path: str | None,
    model_name: str,
    device: torch.device,
    temporal: str = "mean",
    untrained: bool = False,
) -> torch.nn.Module:
    model = build_model(model_name=model_name, temporal=temporal).to(device)
    if untrained:
        model.eval()
        return model
    if not checkpoint_path:
        raise ValueError(f"Checkpoint path required for {model_name} unless --untrained is set.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def iter_videos(path: str, recursive: bool) -> List[str]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            return [str(p)]
        raise ValueError(f"Unsupported file extension: {p.suffix}")
    if not p.is_dir():
        raise FileNotFoundError(path)

    files: List[Path]
    if recursive:
        files = sorted([x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in VIDEO_EXTENSIONS])
    else:
        files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in VIDEO_EXTENSIONS])
    return [str(x) for x in files]


def load_video_ffmpeg(path: str, img_size: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vf",
        f"scale={img_size}:{img_size}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise ValueError(f"ffmpeg failed for {path}: {err[:300]}")
    frame_size = img_size * img_size * 3
    raw = proc.stdout
    if len(raw) < frame_size:
        raise ValueError(f"No frames decoded for {path}")
    n_frames = len(raw) // frame_size
    arr = np.frombuffer(raw[: n_frames * frame_size], dtype=np.uint8)
    return arr.reshape(n_frames, img_size, img_size, 3)


def sample_fixed_clip(frames: np.ndarray, clip_len: int, temporal_stride: int) -> Tuple[np.ndarray, np.ndarray]:
    frames = frames[:: max(1, temporal_stride)]
    total = frames.shape[0]
    if total <= 0:
        raise ValueError("Video has zero frames after temporal stride")

    if total == clip_len:
        idx = np.arange(total, dtype=np.int32)
        return frames, idx
    if total > clip_len:
        start = max(0, (total - clip_len) // 2)
        idx = np.arange(start, start + clip_len, dtype=np.int32)
        return frames[idx], idx

    pad_count = clip_len - total
    pad = np.repeat(frames[-1][None, ...], pad_count, axis=0)
    out = np.concatenate([frames, pad], axis=0)
    idx = np.concatenate([np.arange(total, dtype=np.int32), np.full((pad_count,), total - 1, dtype=np.int32)])
    return out, idx


def normalize_frames(frames: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).contiguous().float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def _leaf_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    out: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if len(list(module.children())) == 0:
            out.append((name, module))
    return out


@torch.no_grad()
def extract_layer_activations(
    model: nn.Module,
    model_name: str,
    frames_tchw: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    activations: Dict[str, np.ndarray] = {}
    hooks = []

    def _make_hook(layer_name: str):
        def _hook(_module: nn.Module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(out, torch.Tensor):
                return
            key = layer_name.replace(".", "__")
            activations[key] = out.detach().to(torch.float32).cpu().numpy()

        return _hook

    for name, module in _leaf_modules(model):
        hooks.append(module.register_forward_hook(_make_hook(name)))

    if model_name == "cnn3d":
        x = frames_tchw.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous().to(device)  # (1, C, T, H, W)
    else:
        x = frames_tchw.unsqueeze(0).to(device)  # (1, T, C, H, W)

    _ = model(x)

    for h in hooks:
        h.remove()

    return activations


def run_one_model(
    model: torch.nn.Module,
    model_name: str,
    videos: List[str],
    out_dir: str,
    device: torch.device,
    clip_len: int,
    img_size: int,
    temporal_stride: int,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    entries: List[Dict[str, object]] = []

    for video_path in videos:
        frames = load_video_ffmpeg(video_path, img_size=img_size)
        clip, _clip_indices = sample_fixed_clip(frames, clip_len=clip_len, temporal_stride=temporal_stride)
        x = normalize_frames(clip)
        out = extract_layer_activations(model=model, model_name=model_name, frames_tchw=x, device=device)

        stem = Path(video_path).stem
        save_path = os.path.join(out_dir, f"{stem}.npz")
        np.savez_compressed(save_path, **out)

        entries.append(
            {
                "video_path": video_path,
                "output_npz": save_path,
                "n_layers": len(out),
            }
        )
        print(f"[{model_name}] {video_path} -> saved {len(out)} layer activations")

    summary: Dict[str, object] = {
        "model_name": model_name,
        "n_videos": len(entries),
        "files": entries,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract time-resolved activations from trained FER 2D/3D checkpoints on stimulus videos."
    )
    parser.add_argument("--input", required=True, help="Stimulus video file or directory")
    parser.add_argument("--checkpoint_2d", default=None, help="Path to best 2D checkpoint (.pt)")
    parser.add_argument("--checkpoint_3d", default=None, help="Path to best 3D checkpoint (.pt)")
    parser.add_argument("--untrained", action="store_true", help="Use randomly initialized models from models.py")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--temporal_2d", default="mean", choices=["mean", "gru"], help="Temporal head of 2D checkpoint")
    parser.add_argument("--clip_len", type=int, default=16, help="Fixed clip length used for extraction")
    parser.add_argument("--img_size", type=int, default=224, help="Resize side for decoded video frames")
    parser.add_argument("--temporal_stride", type=int, default=1, help="Temporal stride before clip sampling")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of videos")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --untrained model initialization. "
                             "Ignored when loading checkpoints. Default 42.")
    args = parser.parse_args()

    # Seed BEFORE building any model so untrained init is deterministic.
    if args.untrained:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    videos = iter_videos(args.input, recursive=args.recursive)
    if args.limit is not None:
        videos = videos[: max(0, args.limit)]
    if len(videos) == 0:
        raise ValueError("No videos found to process.")

    out_dir = os.path.abspath(args.out_dir)
    out_2d = os.path.join(out_dir, "cnn2d")
    out_3d = os.path.join(out_dir, "cnn3d")
    os.makedirs(out_2d, exist_ok=True)
    os.makedirs(out_3d, exist_ok=True)

    if not args.untrained and (not args.checkpoint_2d or not args.checkpoint_3d):
        raise ValueError("Provide --checkpoint_2d and --checkpoint_3d unless --untrained is used.")

    model_2d = load_model(
        args.checkpoint_2d,
        model_name="cnn2d",
        device=device,
        temporal=args.temporal_2d,
        untrained=args.untrained,
    )
    model_3d = load_model(
        args.checkpoint_3d,
        model_name="cnn3d",
        device=device,
        temporal="mean",
        untrained=args.untrained,
    )

    summary_2d = run_one_model(
        model=model_2d,
        model_name="cnn2d",
        videos=videos,
        out_dir=out_2d,
        device=device,
        clip_len=args.clip_len,
        img_size=args.img_size,
        temporal_stride=args.temporal_stride,
    )
    summary_3d = run_one_model(
        model=model_3d,
        model_name="cnn3d",
        videos=videos,
        out_dir=out_3d,
        device=device,
        clip_len=args.clip_len,
        img_size=args.img_size,
        temporal_stride=args.temporal_stride,
    )

    summary = {
        "input": os.path.abspath(args.input),
        "n_videos": len(videos),
        "device": str(device),
        "untrained": args.untrained,
        "clip_len": args.clip_len,
        "img_size": args.img_size,
        "temporal_stride": args.temporal_stride,
        "cnn2d": summary_2d,
        "cnn3d": summary_3d,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
