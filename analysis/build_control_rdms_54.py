#!/usr/bin/env python3
"""Build identity, pixel, motion-energy, and optical-flow controls for 54 stimuli."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "analysis"))
from compute_motion_energy_rdm import build_filter_bank, stim_motion_energy
from compute_optical_flow_rdm import stim_flow_features

OUT = ROOT / "analysis/motion_controls_54/rdms"


def corr_rdm(x):
    x = x - x.mean(1, keepdims=True)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    r = 1 - (x @ x.T) / np.maximum(n * n.T, 1e-12)
    np.fill_diagonal(r, 0)
    return np.nan_to_num(r).astype(np.float32)


def resized_gray(frames):
    out = np.empty((frames.shape[0], frames.shape[1], 128, 128), np.float32)
    for i in range(frames.shape[0]):
        for t in range(frames.shape[1]):
            im = Image.fromarray(frames[i, t]).convert("L").resize((128, 128), Image.BILINEAR)
            out[i, t] = np.asarray(im, np.float32) / 255
    return out


def graded_identity(frames, actors):
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    model = tvm.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 2622)
    model.load_state_dict(torch.load(ROOT / "net_weights/vgg16_vggface",
                                     map_location="cpu", weights_only=False), strict=True)
    feat = nn.Sequential(model.features, model.avgpool, nn.Flatten(),
                         *list(model.classifier[:5])).eval()
    sums = {actor: np.zeros(4096, np.float64) for actor in sorted(set(actors))}
    counts = {actor: 0 for actor in sums}
    with torch.inference_mode():
        for i, actor in enumerate(actors):
            for start in range(0, 16, 4):
                x = frames[i, start:start + 4].astype(np.float32)[..., ::-1].copy()
                x -= np.array([93.5940, 104.7624, 129.1863], np.float32)
                x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy())
                sums[actor] += feat(x).numpy().sum(0)
                counts[actor] += len(x)
            if (i + 1) % 6 == 0:
                print(f"  identity embedding {i + 1}/54", flush=True)
    actor_vectors = {a: sums[a] / counts[a] for a in sums}
    return corr_rdm(np.stack([actor_vectors[a] for a in actors]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "stim_frames_54/stimulus_manifest.json").read_text())
    actors = [row["actor"] for row in manifest]
    frames = np.load(ROOT / "stim_frames_54/all_video_frames_16.npy", mmap_mode="r")

    identity = (np.asarray(actors)[:, None] != np.asarray(actors)[None, :]).astype(np.float32)
    np.fill_diagonal(identity, 0)
    np.save(OUT / "identity_rdm.npy", identity)

    print("[graded identity]")
    np.save(OUT / "identity_graded_rdm.npy", graded_identity(frames, actors))

    print("[resize/grayscale]")
    gray = resized_gray(frames)
    pixel_features = np.abs(gray[:, 1:] - gray[:, :-1]).mean((2, 3))
    np.save(OUT / "pixel_diff_rdm.npy", corr_rdm(pixel_features))

    print("[optical flow]")
    flow = np.stack([stim_flow_features(video) for video in gray])
    np.save(OUT / "optical_flow_rdm.npy", corr_rdm(flow))

    print("[motion energy]")
    bank = build_filter_bank()
    motion = []
    for i, video in enumerate(gray):
        motion.append(stim_motion_energy(video, bank))
        if (i + 1) % 6 == 0:
            print(f"  motion {i + 1}/54", flush=True)
    np.save(OUT / "motion_energy_rdm.npy", corr_rdm(np.log1p(np.stack(motion))))
    for path in sorted(OUT.glob("*.npy")):
        rdm = np.load(path)
        print(f"[ok] {path.name}: {rdm.shape}, finite={np.isfinite(rdm).all()}")


if __name__ == "__main__":
    main()
