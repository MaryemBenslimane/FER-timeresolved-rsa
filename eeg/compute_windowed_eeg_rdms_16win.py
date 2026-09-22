#!/usr/bin/env python3
"""16 non-overlapping EEG RDM windows tiling the FULL epoch.

Each window is 62.5 samples.

Output: <subject>_windowed_rdms.npy of shape (n_roi, 16, 18, 18), plus
roi_order.npy and window_bounds.npy.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

EPS = 1e-12
_ROOT = Path(os.environ.get('FER_ROOT', '.'))
DEFAULT_INPUT_DIR = _ROOT / 'eeg_data' / 'Unmasked_avg'
DEFAULT_OUTPUT_DIR = _ROOT / 'eeg_data' / 'Unmasked_avg_windowed_rdms_16win_full'


def pairwise_distance(x: np.ndarray, metric: str) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        xi = x[i]
        for j in range(i + 1, n):
            xj = x[j]
            if metric == 'euclidean':
                d = float(np.linalg.norm(xi - xj))
            elif metric == 'cosine':
                denom = float(np.linalg.norm(xi) * np.linalg.norm(xj))
                d = 1.0 if denom <= EPS else 1.0 - float(np.dot(xi, xj) / denom)
            else:  # correlation
                xi0 = xi - xi.mean()
                xj0 = xj - xj.mean()
                denom = float(np.linalg.norm(xi0) * np.linalg.norm(xj0))
                d = 1.0 if denom <= EPS else 1.0 - float(np.dot(xi0, xj0) / denom)
            out[i, j] = d
            out[j, i] = d
    return out


def make_windows(n_time: int, n_windows: int) -> np.ndarray:
    """n_windows non-overlapping windows tiling [0, n_time); the last absorbs the remainder."""
    base = n_time // n_windows
    if base < 1:
        raise ValueError(f'n_windows={n_windows} too large for n_time={n_time}')
    bounds = []
    for i in range(n_windows):
        start = i * base
        end = n_time if i == n_windows - 1 else start + base
        bounds.append((start, end))
    return np.array(bounds, dtype=np.int32)


def compute_subject_windowed_rdms(subject_data: Dict[str, np.ndarray], metric: str,
                                  n_windows: int, average_within_window: bool):
    roi_order = sorted(subject_data.keys())
    sample = np.asarray(subject_data[roi_order[0]], dtype=np.float32)
    if sample.ndim != 3:
        raise ValueError(f'Expected ROI arrays (conditions, channels, time), got {sample.shape}')

    n_conditions, _, n_time = sample.shape
    windows = make_windows(n_time, n_windows)
    rdms = np.zeros((len(roi_order), n_windows, n_conditions, n_conditions), dtype=np.float32)

    for roi_idx, roi in enumerate(roi_order):
        arr = np.asarray(subject_data[roi], dtype=np.float32)
        if arr.shape[0] != n_conditions or arr.shape[2] != n_time:
            raise ValueError(f'Inconsistent ROI shape for {roi}: {arr.shape}')
        for wi, (start, end) in enumerate(windows):
            window = arr[:, :, start:end]                       # (cond, ch, win_len)
            if average_within_window:
                features = window.mean(axis=2)
            else:
                features = window.reshape(n_conditions, -1)     # channels x time flattened
            rdms[roi_idx, wi] = pairwise_distance(features, metric=metric)

    return rdms, roi_order, windows


def main() -> None:
    p = argparse.ArgumentParser(description='16 non-overlapping EEG RDM windows covering the full epoch.')
    p.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--metric', choices=['correlation', 'cosine', 'euclidean'], default='correlation')
    p.add_argument('--n-windows', type=int, default=16)
    p.add_argument('--average-within-window', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pkl_files = sorted(args.input_dir.glob('*_avg.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f"No '*_avg.pkl' files in {args.input_dir}")

    shared_rois = None
    shared_windows = None
    for pkl_path in pkl_files:
        subject = pkl_path.stem.replace('_avg', '')
        out_path = args.output_dir / f'{subject}_windowed_rdms.npy'
        if out_path.exists() and not args.overwrite:
            print(f'[skip] {out_path.name}'); continue
        with pkl_path.open('rb') as f:
            data = pickle.load(f)
        rdms, roi_order, windows = compute_subject_windowed_rdms(
            data, metric=args.metric, n_windows=args.n_windows,
            average_within_window=args.average_within_window)
        np.save(out_path, rdms)
        print(f'[ok] {pkl_path.name} -> {out_path.name} {rdms.shape}')
        if shared_rois is None:
            shared_rois, shared_windows = roi_order, windows
        elif roi_order != shared_rois or not np.array_equal(windows, shared_windows):
            raise ValueError('ROI order or window bounds differ across subjects')

    if shared_rois is not None:
        np.save(args.output_dir / 'roi_order.npy', np.array(shared_rois, dtype=object))
        np.save(args.output_dir / 'window_bounds.npy', shared_windows)
        lens = shared_windows[:, 1] - shared_windows[:, 0]
        print(f'[ok] saved roi_order.npy and window_bounds.npy')
        print(f'     window lengths: {lens.tolist()}  (total {int(lens.sum())} samples)')


if __name__ == '__main__':
    main()
