"""Motion-energy RDM via spatio-temporal Gabor filter bank (Adelson & Bergen 1985).

Per stim: convolve grayscale video with a small bank of complex spatio-temporal
Gabor filters (quadrature pairs), take amplitude (sum of squared real/imag),
average over space, then take a temporal summary statistic (mean) per filter.

Feature vector per stim = (n_filters,)  with n_filters = n_orient × n_spatial_freq × n_temporal_freq
                        ≈ 32-64 dimensions.

18 × 18 RDM via correlation distance.

Implementation: 3-D direct convolution via scipy.signal.fftconvolve (no opencv).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _stim_loader_v2 import load_all_stims as _load_v2, N_STIM


def load_all_stims(target_size, grayscale):
    X, _labels = _load_v2(target_size=target_size, grayscale=grayscale)
    return X

OUT = Path(os.environ.get("FER_ROOT",".")+"/analysis/motion_controls/rdms/motion_energy_rdm.npy")


# ── filter-bank construction ───────────────────────────────────────────────
def spatio_temporal_gabor(
    sf: float, tf: float, theta: float,
    ksz_s: int = 11, ksz_t: int = 5,
    sigma_s: float = 2.5, sigma_t: float = 1.2,
) -> np.ndarray:
    """Complex 3-D Gabor: gaussian envelope × complex exponential.

    Returns array of shape (ksz_t, ksz_s, ksz_s), complex128.
        sf      = spatial freq (cycles per pixel)
        tf      = temporal freq (cycles per frame; sign → motion direction)
        theta   = spatial orientation (radians)
        ksz_s   = spatial kernel size (odd)
        ksz_t   = temporal kernel size (odd)
        sigma_s = spatial gaussian sigma (pixels)
        sigma_t = temporal gaussian sigma (frames)
    """
    t  = np.arange(ksz_t) - ksz_t // 2
    y  = np.arange(ksz_s) - ksz_s // 2
    x  = np.arange(ksz_s) - ksz_s // 2
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    # spatial coords along orientation
    xt = xx * np.cos(theta) + yy * np.sin(theta)
    env = np.exp(-(xx**2 + yy**2) / (2 * sigma_s**2)
                 - tt**2 / (2 * sigma_t**2))
    carrier = np.exp(2j * np.pi * (sf * xt + tf * tt))
    g = env * carrier
    # zero-mean (per filter) to remove DC response
    g = g - g.mean()
    return g.astype(np.complex128)


def build_filter_bank() -> list[tuple[str, np.ndarray]]:
    """Return list of (label, complex 3-D kernel)."""
    spatial_freqs  = (0.05, 0.10, 0.20)             # cycles/pixel — coarse to mid
    temporal_freqs = (-0.30, -0.15, 0.15, 0.30)     # cycles/frame, signed = direction
    orientations   = (0.0, np.pi/4, np.pi/2, 3*np.pi/4)
    bank = []
    for sf in spatial_freqs:
        for tf in temporal_freqs:
            for th in orientations:
                lbl = f"sf{sf:.2f}_tf{tf:+.2f}_th{int(np.degrees(th)):03d}"
                bank.append((lbl, spatio_temporal_gabor(sf, tf, th)))
    return bank


# ── motion-energy per stim ────────────────────────────────────────────────
def stim_motion_energy(video: np.ndarray, bank: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """video: (T, H, W) float  → feature vector (n_filters,) of total motion
    energy per filter (summed over space and time)."""
    feat = np.zeros(len(bank), dtype=np.float64)
    for k, (_, kernel) in enumerate(bank):
        # Complex convolution: a Gabor's response decomposes into the real
        # and imag parts of the convolution with the analytic kernel.
        re = fftconvolve(video, kernel.real, mode="same")
        im = fftconvolve(video, kernel.imag, mode="same")
        # amplitude squared (quadrature energy)
        energy = re**2 + im**2
        feat[k] = energy.mean()
    return feat


# ── RDM build ─────────────────────────────────────────────────────────────
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

    bank = build_filter_bank()
    print(f"  filter bank: {len(bank)} filters "
          f"(spatial freqs × temporal freqs × orientations)")
    print(f"  computing motion energy per stim...")

    features = np.zeros((N_STIM, len(bank)), dtype=np.float64)
    for i in range(N_STIM):
        features[i] = stim_motion_energy(X[i], bank)
        if i % 6 == 5:
            print(f"    {i+1}/{N_STIM} stims done")

    # Log-transform to compress the dynamic range (energies span orders of mag)
    features = np.log1p(features)
    print(f"  features: {features.shape}  "
          f"range=[{features.min():.3f}, {features.max():.3f}]")

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
