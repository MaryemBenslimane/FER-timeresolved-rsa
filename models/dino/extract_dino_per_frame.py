#!/usr/bin/env python3
"""Trained DINOv2-temporal PER-FRAME RDMs, re-extracted from best_model.pt.

Rebuilds `layer_rdms_stacked_18x18xframesxlayers.npy` (whose producing script is no
longer in the tree) using the VERIFIED machinery already in `rdms_compute.py`: the
same model class, the same hooked-layer map, the same activation reshaping. Only the
RDM stage differs -- here one correlation-distance RDM is built per (layer, frame)
instead of one per layer.

Layers: the 19 that carry a per-frame representation -- 12 DINO blocks, dino_norm,
frame_proj, pos_enc, temporal_transformer x2, temporal_norm, frame_classifier.
`video_classifier` is excluded: it emits one vector per video, so it has no frame axis
(matching `analysis/model_prep.prepare_dinov2_temporal`).

Stimulus order = sorted video filenames, which -- now that the Hapiness_Actor07 typo is
fixed -- is the canonical emotion-major / numeric-actor order.

Output: (n_layer, n_frames, 18, 18) float32 + a sidecar .json with the layer order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rdms_compute import (  # verified components, reused as-is
    DinoV2TemporalTransformer,
    StimulusVideoDataset,
    collate_stimulus_videos,
    build_layer_module_map,
    LayerActivationCollector,
    reshape_layer_activation,
)

EXCLUDE = {"video_classifier"}          # no per-frame axis


def per_frame_vectors(layer_name: str, act: np.ndarray, num_frames: int) -> np.ndarray:
    """(B, ...) activation -> (B, num_frames, D) one vector per stimulus per frame."""
    if act.ndim == 4:                    # (B, T, P, D) dino blocks: mean over patches
        return act.mean(axis=2)
    if act.ndim == 3:
        if act.shape[1] == num_frames + 1:   # (B, T+1, D): CLS at 0, frames at 1:
            return act[:, 1:, :]
        if act.shape[1] == num_frames:       # (B, T, D)
            return act
    raise ValueError(f"{layer_name}: cannot extract a frame axis from shape {act.shape}")


def correlation_rdm(x: np.ndarray) -> np.ndarray:
    """x: (N, D) -> (N, N) correlation distance."""
    corr = np.corrcoef(x)
    rdm = 1.0 - corr
    np.fill_diagonal(rdm, 0.0)
    return np.nan_to_num(rdm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/home/maryem/scratch/FER/DINOv2/best_model.pt")
    ap.add_argument("--stimulus-dir", default="/home/maryem/scratch/FER/Stims_Videos/Stims_Videos")
    ap.add_argument("--out", default="/home/maryem/scratch/FER/analysis/model_rdms_canonical/dinov2_temporal_canon.npy")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    class_to_idx = ckpt.get("class_to_idx", None)
    num_classes = cfg.get("num_classes", len(class_to_idx) if class_to_idx else 3)
    num_frames = cfg.get("num_frames", args.num_frames)
    proj_dim = cfg.get("proj_dim", 256)
    image_size = cfg.get("image_size", args.image_size)
    print(f"[ckpt] num_frames={num_frames} proj_dim={proj_dim} num_classes={num_classes}")

    model = DinoV2TemporalTransformer(
        num_classes=num_classes, num_frames=num_frames, dino_model_name="dinov2_vitb14",
        proj_dim=proj_dim, num_layers=2, num_heads=8, mlp_ratio=4.0, dropout=0.2,
        freeze_dino=True,
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True), None
    model.eval()
    print(f"[model] loaded {args.checkpoint} (strict=True)")

    modules = {k: v for k, v in build_layer_module_map(model).items() if k not in EXCLUDE}
    layer_order = list(modules.keys())
    print(f"[hooks] {len(layer_order)} layers: {layer_order}")
    collector = LayerActivationCollector(modules)

    ds = StimulusVideoDataset(root_dir=args.stimulus_dir, num_frames=num_frames, image_size=image_size)
    print(f"[data] {len(ds)} stimuli (sorted filename order):")
    for i, s in enumerate(ds.samples):
        print(f"   [{i:2d}] {Path(s).name}")
    loader = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=collate_stimulus_videos)

    feats = {k: [None] * len(ds) for k in layer_order}
    for batch in loader:
        video = batch["video"].to(device)
        idxs = batch["index"].tolist()
        B, T = video.shape[0], video.shape[1]
        collector.clear()
        with torch.no_grad():
            model(video)
        for k in layer_order:
            act = reshape_layer_activation(k, collector.current_batch[k], B, T)
            pf = per_frame_vectors(k, act, T)                  # (B, T, D)
            pf = pf.reshape(B, T, -1)
            for b in range(B):
                feats[k][idxs[b]] = pf[b]

    out = np.zeros((len(layer_order), num_frames, len(ds), len(ds)), dtype=np.float32)
    for li, k in enumerate(layer_order):
        stack = np.stack(feats[k], 0)                          # (18, T, D)
        for t in range(num_frames):
            out[li, t] = correlation_rdm(stack[:, t, :])
    collector.remove()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    Path(args.out).with_suffix(".layers.json").write_text(json.dumps(
        {"layer_order": layer_order, "shape": list(out.shape),
         "stimuli": [Path(s).name for s in ds.samples]}, indent=2))
    print(f"[ok] {args.out}  shape={out.shape}  finite={bool(np.isfinite(out).all())}")


if __name__ == "__main__":
    main()
