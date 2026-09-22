"""Pixel-difference RDM — simplest motion baseline.

For each stim:
  diff_t = mean over space of |frame[t+1] - frame[t]|, for t = 0..13
→ 14-dim feature vector per stim
→ 18x18 RDM via correlation distance

This is a sanity baseline for the motion-energy and optical-flow RDMs.
Distinct from analysis/pixel_rdm_control.py (which uses STATIC pooled pixels,
not differences).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _stim_loader_v2 import load_all_stims as _load_v2, N_STIM


def load_all_stims(target_size, grayscale):
    X, _labels = _load_v2(target_size=target_size, grayscale=grayscale)
    return X

OUT = Path(os.environ.get("FER_ROOT",".")+"/analysis/motion_controls/rdms/pixel_diff_rdm.npy")


def correlation_distance_rdm(features: np.ndarray) -> np.ndarray:
    Xc    = features - features.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    denom = np.maximum(norms * norms.T, 1e-12)
    cor   = (Xc @ Xc.T) / denom
    rdm   = 1.0 - cor
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def build() -> np.ndarray:
    print("Loading stim frames (grayscale, 128x128)...")
    X = load_all_stims(target_size=(128, 128), grayscale=True)  # (18, 15, H, W)
    assert X.shape[0] == N_STIM
    print(f"  stim tensor: {X.shape}")

    # Per-stim feature: 14-d (one per frame-transition), each = mean |Δ| over space
    diffs = np.abs(X[:, 1:] - X[:, :-1])              # (18, 14, H, W)
    features = diffs.mean(axis=(2, 3))                 # (18, 14)
    print(f"  features per stim: {features.shape}  "
          f"range=[{features.min():.4f}, {features.max():.4f}]")

    rdm = correlation_distance_rdm(features)
    print(f"  RDM shape={rdm.shape}  range=[{rdm.min():.3f}, {rdm.max():.3f}]")
    return rdm


def main() -> None:
    rdm = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, rdm)
    print(f"\n[ok] saved → {OUT}")


if __name__ == "__main__":
    main()
