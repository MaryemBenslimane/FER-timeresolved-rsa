import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def iter_npz_files(path: str) -> List[str]:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"Activation directory not found: {path}")
    files = sorted([str(x) for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".npz"])
    if not files:
        raise ValueError(f"No .npz activation files found in: {path}")
    return files


def pairwise_distance(x: np.ndarray, metric: str) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        xi = x[i]
        for j in range(i + 1, n):
            xj = x[j]
            if metric == "euclidean":
                d = float(np.linalg.norm(xi - xj))
            elif metric == "cosine":
                denom = float(np.linalg.norm(xi) * np.linalg.norm(xj))
                d = 1.0 if denom == 0 else 1.0 - float(np.dot(xi, xj) / denom)
            else:  # correlation
                xi0 = xi - xi.mean()
                xj0 = xj - xj.mean()
                denom = float(np.linalg.norm(xi0) * np.linalg.norm(xj0))
                d = 1.0 if denom == 0 else 1.0 - float(np.dot(xi0, xj0) / denom)
            out[i, j] = d
            out[j, i] = d
    return out


def _infer_time_axis(arr: np.ndarray, expected_timepoints: int) -> int:
    if arr.ndim == 0:
        return -1
    # Drop batch dim if present.
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        return -1

    candidates = [i for i, s in enumerate(arr.shape) if s > 1]
    if not candidates:
        return -1

    # Choose axis whose size is closest to requested timepoints.
    best = min(candidates, key=lambda i: abs(arr.shape[i] - expected_timepoints))
    return best


def _resample_time(x_td: np.ndarray, timepoints: int) -> np.ndarray:
    t_in, d = x_td.shape
    if t_in == timepoints:
        return x_td
    if t_in <= 1:
        return np.repeat(x_td, timepoints, axis=0)

    src = np.linspace(0.0, 1.0, t_in, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, timepoints, dtype=np.float32)
    out = np.zeros((timepoints, d), dtype=np.float32)
    for j in range(d):
        out[:, j] = np.interp(dst, src, x_td[:, j]).astype(np.float32)
    return out


def _to_time_by_feature(arr: np.ndarray, timepoints: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]

    axis = _infer_time_axis(x, expected_timepoints=timepoints)
    if axis < 0:
        return np.zeros((timepoints, 1), dtype=np.float32)

    xt = np.moveaxis(x, axis, 0)
    xt = xt.reshape(xt.shape[0], -1).astype(np.float32)
    return _resample_time(xt, timepoints=timepoints)


def _load_stimulus_layer_vectors(
    npz_files: List[str],
    timepoints: int,
) -> Tuple[List[str], List[str], Dict[str, List[np.ndarray]]]:
    stimulus = [Path(p).stem for p in npz_files]
    first = np.load(npz_files[0], allow_pickle=True)
    layer_names = sorted(first.files)
    per_layer: Dict[str, List[np.ndarray]] = {k: [] for k in layer_names}

    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        for layer in layer_names:
            if layer not in data:
                raise ValueError(f"Layer '{layer}' missing in {path}")
            per_layer[layer].append(_to_time_by_feature(data[layer], timepoints=timepoints))

    return stimulus, layer_names, per_layer


def compute_rdms(
    model_dir: str,
    timepoints: int,
    metric: str,
) -> np.ndarray:
    npz_files = iter_npz_files(model_dir)
    _stimulus, layer_names, per_layer = _load_stimulus_layer_vectors(npz_files, timepoints=timepoints)

    n_layers = len(layer_names)
    n_stimulus = len(npz_files)
    rdms = np.zeros((n_layers, timepoints, n_stimulus, n_stimulus), dtype=np.float32)

    for li, layer in enumerate(layer_names):
        layer_stim = per_layer[layer]  # list of (T, D)
        stack = np.stack(layer_stim, axis=0)  # (N, T, D)
        for t in range(timepoints):
            rdms[li, t] = pairwise_distance(stack[:, t, :], metric=metric)

    return rdms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute layer-by-time RDMs from activation .npz files."
    )
    parser.add_argument("--root", required=True, help="Root dir containing model subdirs (e.g., cnn2d, cnn3d)")
    parser.add_argument("--models", nargs="+", default=["cnn2d", "cnn3d"], help="Model subdirs to process")
    parser.add_argument("--timepoints", type=int, required=True, help="Output timepoints in RDMs")
    parser.add_argument("--metric", choices=["correlation", "cosine", "euclidean"], default="correlation")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for model_name in args.models:
        model_dir = os.path.join(args.root, model_name)
        rdms = compute_rdms(model_dir=model_dir, timepoints=args.timepoints, metric=args.metric)
        out_path = os.path.join(args.out_dir, f"rdms_{model_name}.npy")
        np.save(out_path, rdms)
        print(f"[{model_name}] saved {out_path} with shape {rdms.shape}")


if __name__ == "__main__":
    main()
