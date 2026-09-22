#!/usr/bin/env python3
"""Grid-wide cluster permutation tests over layer × time RSA grids.

Missing data
------------
`rsa_full_grid*.py` writes NaN for a participant x condition whose EEG RDM is
degenerate (flat). In the 54-stimulus set 11 of 25 participants are entirely
NaN in the `neutral` and pooled `all` conditions. Such participants are dropped
**listwise per (scope, emotion) grid** and the realised n is reported; they are
never zero-filled, which would shrink the mean toward zero and claim a df the
data do not support. n therefore varies between grids, so the sign matrix and
the cluster-forming t threshold are built per realised n.
"""
import csv
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.stats import t as tdist

ROOT = Path(".")
GRID_DIR = ROOT / "analysis/rsa_grid_54"
OUT = ROOT / "analysis/cluster_permutation_54"
GRID_DIRS_18 = [
    ROOT / "analysis/rsa_grid",
    ROOT / "analysis/rsa_grid_today",
    ROOT / "analysis/rsa_grid_facenet_original_vggface2_affectnet_16win",
]
N_PERM = 2000
SEED = 42
ALPHA = .05
MIN_SUBJECTS = 3
STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def t_values(x, signed_sum=None):
    n = x.shape[0]
    ss = np.sum(x * x, axis=0)
    total = np.sum(x, axis=0) if signed_sum is None else signed_sum
    mean = total / n
    var = np.maximum((ss - n * mean * mean) / (n - 1), 0)
    return np.divide(mean, np.sqrt(var / n), out=np.zeros_like(mean),
                     where=var > 0)


def clusters(tmap, threshold, valid=None):
    hit = tmap > threshold
    if valid is not None:
        hit &= valid
    labels, count = label(hit, structure=STRUCTURE)
    result = []
    for cluster_id in range(1, count + 1):
        mask = labels == cluster_id
        result.append((mask, float(tmap[mask].sum())))
    return result


def max_cluster_null(x, signs, threshold, valid=None):
    n, layers, windows = x.shape
    flat = x.reshape(n, -1)
    ss = np.sum(flat * flat, axis=0)
    null = np.zeros(len(signs), dtype=np.float32)
    for start in range(0, len(signs), 100):
        totals = signs[start:start + 100] @ flat
        means = totals / n
        var = np.maximum((ss[None, :] - n * means * means) / (n - 1), 0)
        tmaps = np.divide(means, np.sqrt(var / n),
                          out=np.zeros_like(means), where=var > 0)
        for j, tflat in enumerate(tmaps):
            found = clusters(tflat.reshape(layers, windows), threshold, valid)
            null[start + j] = max((mass for _, mass in found), default=0)
    return null


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(running, 1.)
    return adjusted


def signs_for(n, cache={}):
    """Sign matrix for a realised n. Seeded per n, so a grid's test is
    reproducible and a complete grid gives exactly the result it gave when the
    matrix was built once for the full sample."""
    if n not in cache:
        rng = np.random.default_rng(SEED)
        cache[n] = rng.choice(np.array([-1., 1.], np.float32), size=(N_PERM, n))
    return cache[n]


def threshold_for(n, cache={}):
    if n not in cache:
        cache[n] = float(tdist.ppf(1 - ALPHA, n - 1))
    return cache[n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["18", "54"], default="54")
    args = parser.parse_args()
    global OUT
    if args.study == "54":
        paths = sorted(GRID_DIR.glob("*_rsa_grid.npz"))
    else:
        OUT = ROOT / "analysis/cluster_permutation_18"
        # Later directories in this list only fill missing tags; never test a
        # duplicated model twice.
        by_tag = {}
        for directory in GRID_DIRS_18:
            for path in sorted(directory.glob("*_rsa_grid.npz")):
                by_tag.setdefault(path.name.removesuffix("_rsa_grid.npz"), path)
        paths = [by_tag[tag] for tag in sorted(by_tag)]
    OUT.mkdir(parents=True, exist_ok=True)
    grid_rows, cluster_rows = [], []

    for pi, path in enumerate(paths, 1):
        tag = path.name.removesuffix("_rsa_grid.npz")
        z = np.load(path, allow_pickle=True)
        rsa = np.asarray(z["rsa"], np.float32)
        scopes = [str(x) for x in z["scopes"]]
        emotions = [str(x) for x in z["emotions"]]
        subjects = ([str(x) for x in z["subjects"]] if "subjects" in z
                    else [str(i) for i in range(rsa.shape[0])])
        model_rows = []
        masks = np.zeros(rsa.shape[1:], dtype=np.uint8)  # scope, layer, emotion, window
        for si, scope in enumerate(scopes):
            for ei, emotion in enumerate(emotions):
                raw = rsa[:, si, :, ei, :]
                finite = np.isfinite(raw).reshape(raw.shape[0], -1)
                keep = finite.any(axis=1)
                n = int(keep.sum())
                dropped = [subjects[i] for i in np.flatnonzero(~keep)]
                base = {
                    "model": tag, "scope": scope, "emotion": emotion,
                    "n": n, "n_dropped": len(dropped),
                    "dropped_subjects": ";".join(dropped),
                }
                if n < MIN_SUBJECTS:
                    model_rows.append({
                        **base, "n_layers": raw.shape[1],
                        "n_windows": raw.shape[2], "t_threshold": float("nan"),
                        "n_permutations": 0, "n_valid_cells": 0,
                        "n_cells": raw.shape[1] * raw.shape[2],
                        "n_clusters": 0, "max_t": float("nan"),
                        "min_cluster_p": 1.0, "survives_grid_fwer": 0,
                    })
                    continue
                x = raw[keep]
                # A cell enters clustering only if every retained participant
                # has it; zero-filling here touches no retained cell.
                valid = np.isfinite(x).all(axis=0)
                x = np.where(np.isfinite(x), x, 0.).astype(np.float32)
                threshold = threshold_for(n)
                signs = signs_for(n)
                observed_t = np.where(valid, t_values(x), 0.)
                observed_clusters = clusters(observed_t, threshold, valid)
                null = max_cluster_null(x, signs, threshold, valid)
                cluster_ps = []
                for ci, (mask, mass) in enumerate(observed_clusters, 1):
                    p = float((1 + np.sum(null >= mass)) / (N_PERM + 1))
                    cluster_ps.append(p)
                    if p < ALPHA:
                        masks[si, :, ei, :][mask] = 1
                    where = np.argwhere(mask)
                    cluster_rows.append({
                        **base,
                        "cluster": ci, "mass": mass, "size": int(mask.sum()),
                        "layer_min": int(where[:, 0].min()),
                        "layer_max": int(where[:, 0].max()),
                        "window_min": int(where[:, 1].min()),
                        "window_max": int(where[:, 1].max()),
                        "peak_t": float(observed_t[mask].max()),
                        "mean_rho": float(np.nanmean(raw[keep][:, mask])),
                        "p_grid_fwer": p,
                    })
                model_rows.append({
                    **base,
                    "n_layers": x.shape[1], "n_windows": x.shape[2],
                    "t_threshold": threshold, "n_permutations": N_PERM,
                    "n_valid_cells": int(valid.sum()),
                    "n_cells": int(valid.size),
                    "n_clusters": len(observed_clusters),
                    "max_t": float(observed_t[valid].max()) if valid.any() else float("nan"),
                    "min_cluster_p": min(cluster_ps, default=1.0),
                    "survives_grid_fwer": int(any(p < ALPHA for p in cluster_ps)),
                })
        adjusted = holm([row["min_cluster_p"] for row in model_rows])
        for row, adj in zip(model_rows, adjusted):
            row["p_holm_28_grids"] = float(adj)
            row["survives_holm_28_grids"] = int(adj < ALPHA)
        grid_rows.extend(model_rows)
        shape = (len(scopes), len(emotions))
        np.savez_compressed(
            OUT / f"{tag}_cluster_masks.npz", significant=masks,
            scopes=np.asarray(scopes), emotions=np.asarray(emotions),
            # n and the t threshold now vary per grid with listwise deletion.
            n=np.asarray([r["n"] for r in model_rows], np.int16).reshape(shape),
            threshold=np.asarray([r["t_threshold"] for r in model_rows],
                                 np.float64).reshape(shape),
            n_permutations=N_PERM,
        )
        print(f"[{pi}/{len(paths)}] {tag}: "
              f"{sum(r['survives_grid_fwer'] for r in model_rows)} grid-FWER, "
              f"{sum(r['survives_holm_28_grids'] for r in model_rows)} Holm, "
              f"n={sorted(set(r['n'] for r in model_rows))}", flush=True)

    with (OUT / "grid_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grid_rows[0].keys())
        writer.writeheader(); writer.writerows(grid_rows)
    with (OUT / "clusters.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cluster_rows[0].keys())
        writer.writeheader(); writer.writerows(cluster_rows)

    survivors = []
    for tag in sorted(set(row["model"] for row in grid_rows)):
        rows = [row for row in grid_rows if row["model"] == tag]
        survivors.append({
            "model": tag,
            "n_min": min(r["n"] for r in rows),
            "n_max": max(r["n"] for r in rows),
            "n_grid_fwer_survivors": sum(r["survives_grid_fwer"] for r in rows),
            "n_holm_survivors": sum(r["survives_holm_28_grids"] for r in rows),
            "model_survives_grid_fwer": int(any(r["survives_grid_fwer"] for r in rows)),
            "model_survives_holm": int(any(r["survives_holm_28_grids"] for r in rows)),
            "best_cluster_p": min(r["min_cluster_p"] for r in rows),
            "best_holm_p": min(r["p_holm_28_grids"] for r in rows),
        })
    with (OUT / "model_survivors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=survivors[0].keys())
        writer.writeheader(); writer.writerows(survivors)
    (OUT / "method.json").write_text(json.dumps({
        "test": "one-sample one-sided sign-flip",
        "grid": "full model-layer x EEG-window",
        "adjacency": "4-neighbour",
        "cluster_statistic": "sum of t",
        "cluster_forming_alpha": ALPHA,
        "within_grid_correction": "maximum cluster mass FWER",
        "n_permutations": N_PERM, "seed": SEED,
        "across_grid_correction": "Holm across 7 scopes x 4 emotions per model",
        "missing_data": ("listwise deletion per (scope, emotion) grid; "
                         "participants with no finite cell are dropped and the "
                         "realised n is reported per grid, from which the sign "
                         "matrix and the t threshold are derived"),
        "min_subjects_tested": MIN_SUBJECTS,
    }, indent=2))
    print(f"[ok] -> {OUT}")


if __name__ == "__main__":
    main()
