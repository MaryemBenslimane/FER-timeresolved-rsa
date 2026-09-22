"""Load the 18 canonical stimuli as videos, for the motion-control RDMs.

stim_frames/Slide{k}.h5['images'] is (18, 224, 224, 3) uint8 RGB: slide k holds
FRAME k of all 18 stimuli. This transposes that into per-stimulus videos.

    load_all_stims(target_size=(128,128), grayscale=True) -> (X, labels)
      X      (18, n_frames, H, W)  float32 in [0,1]   (grayscale)
             (18, n_frames, H, W, 3)                  (colour)
      labels list of 18 stimulus names, canonical emotion-major order

Canonical order is EMOTION-MAJOR: fear 0-5, happy 6-11, neutral 12-17, each
block running over actors 01, 02, 04, 07, 08, 11.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("FER_ROOT", "."))
STIM_DIR = Path(os.environ.get("STIM_DIR", ROOT / "stim_frames"))

N_STIM = 18
ACTORS = [1, 2, 4, 7, 8, 11]
STIMS = [f"{e}_Actor{a:02d}" for e in ("fear", "happy", "neutral") for a in ACTORS]


def n_frames(stim_dir: Path = STIM_DIR) -> int:
    return len(sorted(stim_dir.glob("Slide*.h5")))


def load_all_stims(target_size=(128, 128), grayscale: bool = True,
                   stim_dir: Path = STIM_DIR):
    n_f = n_frames(stim_dir)
    if n_f == 0:
        raise FileNotFoundError(f"no Slide*.h5 in {stim_dir}")

    per_frame = []
    for k in range(1, n_f + 1):
        with h5py.File(stim_dir / f"Slide{k}.h5", "r") as f:
            img = np.asarray(f["images"])                  # (18, H, W, 3) uint8
        if img.shape[0] != N_STIM:
            raise ValueError(f"Slide{k}.h5 has {img.shape[0]} stimuli, expected {N_STIM}")
        if grayscale:
            # ITU-R 601-2 luminance, matching PIL's RGB->L
            img = (img[..., 0] * 0.299 + img[..., 1] * 0.587
                   + img[..., 2] * 0.114).astype(np.float32)
        else:
            img = img.astype(np.float32)

        if target_size is not None and img.shape[1:3] != tuple(target_size):
            mode = "F" if grayscale else "RGB"
            out = []
            for s in range(N_STIM):
                a = img[s]
                if grayscale:
                    out.append(np.array(Image.fromarray(a, mode=mode)
                                        .resize(target_size[::-1], Image.BILINEAR)))
                else:
                    out.append(np.array(Image.fromarray(a.astype(np.uint8), mode=mode)
                                        .resize(target_size[::-1], Image.BILINEAR),
                                        dtype=np.float32))
            img = np.stack(out, 0)
        per_frame.append(img)

    X = np.stack(per_frame, axis=1) / 255.0                # (18, n_frames, H, W[, 3])
    return X.astype(np.float32), list(STIMS)


if __name__ == "__main__":
    X, labels = load_all_stims()
    print(f"X {X.shape} {X.dtype} range=[{X.min():.3f},{X.max():.3f}]")
    print(f"{len(labels)} labels, first 3: {labels[:3]}")
