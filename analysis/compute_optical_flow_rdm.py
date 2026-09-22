"""Optical-flow RDM via simple gradient-based dense flow (no opencv).

For each consecutive frame pair we estimate a coarse dense flow field using a
Horn–Schunck-style smoothing of the brightness-constancy equation, computed
analytically per pixel from the local image gradients (Lucas-Kanade with a
larger window).

Per stim → flow magnitude histogram (16 bins) × flow direction histogram (8 bins)
concatenated across the 14 frame transitions
→ 14 × 24 = 336-dim feature vector
→ 18 × 18 RDM via correlation distance.

Implementation uses scipy.ndimage and numpy — no opencv dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _stim_loader_v2 import load_all_stims as _load_v2, N_STIM


def load_all_stims(target_size, grayscale):
    X, _labels = _load_v2(target_size=target_size, grayscale=grayscale)
    return X

OUT = Path(os.environ.get("FER_ROOT",".")+"/analysis/motion_controls/rdms/optical_flow_rdm.npy")

MAG_BINS  = 16
DIR_BINS  = 8
WINDOW_PX = 7   # Lucas-Kanade window
SIGMA_PRE = 1.0


def lucas_kanade_flow(I1: np.ndarray, I2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense Lucas-Kanade flow between two grayscale frames.
    Returns (u, v) of same shape; uv = flow vector at each pixel.
    """
    I1s = gaussian_filter(I1, SIGMA_PRE)
    I2s = gaussian_filter(I2, SIGMA_PRE)
    Ix = 0.5 * (np.roll(I1s, -1, axis=1) - np.roll(I1s, 1, axis=1))
    Iy = 0.5 * (np.roll(I1s, -1, axis=0) - np.roll(I1s, 1, axis=0))
    It = I2s - I1s

    # Aggregate the LK system over the local window:
    # [ Σ Ix²    Σ IxIy ] [u]   [ −Σ IxIt ]
    # [ Σ IxIy  Σ Iy²  ] [v] = [ −Σ IyIt ]
    a = uniform_filter(Ix * Ix, WINDOW_PX)
    b = uniform_filter(Ix * Iy, WINDOW_PX)
    d = uniform_filter(Iy * Iy, WINDOW_PX)
    e = uniform_filter(Ix * It, WINDOW_PX)
    f = uniform_filter(Iy * It, WINDOW_PX)
    det = a * d - b * b
    det = np.where(np.abs(det) < 1e-10, 1e-10, det)
    u = (-d * e + b * f) / det
    v = ( b * e - a * f) / det
    return u, v


def flow_histograms(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Histograms of (magnitude, direction). Returns 24-dim vector."""
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)  # range (-π, π]
    # mask out near-zero flow (uninformative)
    valid = mag > np.percentile(mag, 5)
    mag_v, ang_v = mag[valid], ang[valid]
    if len(mag_v) == 0:
        return np.zeros(MAG_BINS + DIR_BINS, dtype=np.float64)
    mag_max = np.percentile(mag_v, 99) + 1e-6
    h_mag, _ = np.histogram(np.clip(mag_v, 0, mag_max),
                            bins=MAG_BINS, range=(0, mag_max), density=True)
    h_dir, _ = np.histogram(ang_v, bins=DIR_BINS, range=(-np.pi, np.pi), density=True)
    return np.concatenate([h_mag, h_dir]).astype(np.float64)


def stim_flow_features(video: np.ndarray) -> np.ndarray:
    """video: (T, H, W) → feature vector  ((T-1) * (MAG_BINS + DIR_BINS),)."""
    T = video.shape[0]
    feats = []
    for t in range(T - 1):
        u, v = lucas_kanade_flow(video[t], video[t + 1])
        feats.append(flow_histograms(u, v))
    return np.concatenate(feats, axis=0)


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
    X = load_all_stims(target_size=(128, 128), grayscale=True)
    assert X.shape[0] == N_STIM
    print(f"  stim tensor: {X.shape}")

    feat_dim = (X.shape[1] - 1) * (MAG_BINS + DIR_BINS)
    print(f"  computing dense LK flow per stim → {feat_dim}-dim feature vector")
    features = np.zeros((N_STIM, feat_dim), dtype=np.float64)
    for i in range(N_STIM):
        features[i] = stim_flow_features(X[i])
        if i % 6 == 5:
            print(f"    {i+1}/{N_STIM} stims done")

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
