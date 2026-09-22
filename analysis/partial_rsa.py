#!/usr/bin/env python3
"""Time-locked partial Spearman RSA for the 18- and 54-stimulus analyses.

For each participant, scalp scope, model layer, condition, and matched time bin,
this script rank-transforms the upper triangles of the model and EEG RDMs,
residualizes both vectors against ranked nuisance RDM vectors, and correlates
the residuals. It computes four single-control analyses (actor identity, pixels,
motion energy, and optical flow) and one joint-control analysis.

The joint participant-level grids are retained for full-grid statistical tests.
The CSV contains descriptive maxima extracted only after participant averaging.
ROI-avg is the mean of the six independently selected ROI-specific maxima.

Outputs:
  analysis/partial_rsa_18/<model>_partial_joint_grid.npz
  analysis/partial_rsa_18/partial_rsa_maxima.csv
  analysis/partial_rsa_54/<model>_partial_joint_grid.npz
  analysis/partial_rsa_54/partial_rsa_maxima.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CONTROL_NAMES = ("identity", "pixel_diff", "motion_energy", "optical_flow")
EMOTIONS = ("all", "fear", "happy", "neutral")


def upper(x: np.ndarray) -> np.ndarray:
    ij = np.triu_indices(x.shape[-1], 1)
    return x[..., ij[0], ij[1]]


def ranked_centered(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x, axis=-1)
    return ranks - ranks.mean(axis=-1, keepdims=True)


def nuisance_basis(control_vectors: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Return an orthonormal basis for nonconstant, nonredundant ranked controls."""
    columns, retained = [], []
    for index, vector in enumerate(control_vectors):
        ranked = ranked_centered(np.asarray(vector)[None])[0]
        if np.linalg.norm(ranked) > 1e-12:
            columns.append(ranked)
            retained.append(index)
    if not columns:
        return np.empty((len(control_vectors[0]), 0)), []
    design = np.stack(columns, axis=1)
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = np.finfo(float).eps * max(design.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    return u[:, :rank], retained


def residual_normalise(vectors: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = ranked_centered(vectors)
    if basis.shape[1]:
        values = values - (values @ basis) @ basis.T
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    invalid = norm[..., 0] < 1e-12
    values = np.divide(values, np.clip(norm, 1e-12, None))
    return values, invalid


def subject_ids_18() -> list[str]:
    directory = ROOT / "eeg_data/Unmasked_avg_windowed_rdms_16win_full"
    complete = set(subject_ids_54())
    return sorted(p.name.removesuffix("_windowed_rdms.npy") for p in directory.glob("*_windowed_rdms.npy")
                  if not p.name.startswith("group") and
                  p.name.removesuffix("_windowed_rdms.npy") in complete)


def load_eeg_18(emotion: str, subjects: list[str]) -> tuple[np.ndarray, list[str]]:
    roi_dir = ROOT / "eeg_data/Unmasked_avg_windowed_rdms_16win_full"
    ws_dir = ROOT / "eeg_data/Unmasked_avg_wholescalp_16win"
    roi = np.stack([np.load(roi_dir / f"{s}_windowed_rdms.npy") for s in subjects])
    ws = np.stack([np.load(ws_dir / f"{s}_wholescalp_rdms.npy") for s in subjects])
    scopes = [str(x) for x in np.load(roi_dir / "roi_order.npy", allow_pickle=True)] + ["wholescalp"]
    eeg = np.concatenate([roi, ws[:, None]], axis=1)
    indices = {
        "all": np.arange(18), "fear": np.arange(6),
        "happy": np.arange(6, 12), "neutral": np.arange(12, 18),
    }[emotion]
    return eeg[..., indices, :][..., indices], scopes


def subject_ids_54() -> list[str]:
    directory = ROOT / "eeg_rdms_complete/perroi"
    return sorted(p.name.removesuffix("_all_rdms.npy") for p in directory.glob("*_all_rdms.npy"))


def load_eeg_54(emotion: str, subjects: list[str]) -> tuple[np.ndarray, list[str]]:
    base = ROOT / "eeg_rdms_complete"
    roi = np.stack([np.load(base / "perroi" / f"{s}_{emotion}_rdms.npy") for s in subjects])
    ws = np.stack([np.load(base / "wholescalp" / f"{s}_{emotion}_rdms.npy") for s in subjects])
    scopes = [str(x) for x in np.load(base / "roi_order.npy", allow_pickle=True)] + ["wholescalp"]
    return np.concatenate([roi, ws[:, None]], axis=1), scopes


def condition_indices(dataset: int, emotion: str) -> np.ndarray:
    size = 6 if dataset == 18 else 18
    return {
        "all": np.arange(size * 3), "fear": np.arange(size),
        "happy": np.arange(size, size * 2), "neutral": np.arange(size * 2, size * 3),
    }[emotion]


def model_tags(dataset: int) -> list[str]:
    grid_dir = ROOT / ("analysis/rsa_grid" if dataset == 18 else "analysis/rsa_grid_54")
    return sorted(p.name.removesuffix("_rsa_grid.npz") for p in grid_dir.glob("*_rsa_grid.npz"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, choices=(18, 54), required=True)
    parser.add_argument("--models", nargs="*", help="optional model RDM basenames")
    args = parser.parse_args()

    dataset = args.dataset
    subjects = subject_ids_18() if dataset == 18 else subject_ids_54()
    model_dir = ROOT / ("analysis/model_rdms_canonical" if dataset == 18 else "analysis/model_rdms_54")
    control_dir = ROOT / ("analysis/motion_controls/rdms" if dataset == 18 else
                          "analysis/motion_controls_54/rdms")
    output_dir = ROOT / f"analysis/partial_rsa_{dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tags = args.models or model_tags(dataset)
    controls = {name: np.load(control_dir / f"{name}_rdm.npy") for name in CONTROL_NAMES}
    rows: list[dict[str, object]] = []

    for tag in tags:
        path = model_dir / f"{tag}.npy"
        if not path.exists():
            print(f"[skip] missing {path}", flush=True)
            continue
        model = np.load(path)
        joint_by_emotion = []
        scopes_out = None

        for emotion in EMOTIONS:
            eeg, scopes = (load_eeg_18(emotion, subjects) if dataset == 18 else
                           load_eeg_54(emotion, subjects))
            scopes_out = scopes
            indices = condition_indices(dataset, emotion)
            model_condition = model[..., indices, :][..., indices]
            model_vectors = upper(model_condition)
            eeg_vectors = upper(eeg)
            control_vectors = [upper(controls[name][np.ix_(indices, indices)])
                               for name in CONTROL_NAMES]
            control_sets = [(name, [i]) for i, name in enumerate(CONTROL_NAMES)]
            control_sets.append(("joint", list(range(len(CONTROL_NAMES)))))

            for set_name, selected in control_sets:
                basis, retained_local = nuisance_basis([control_vectors[i] for i in selected])
                retained = [CONTROL_NAMES[selected[i]] for i in retained_local]
                ez, e_invalid = residual_normalise(eeg_vectors, basis)
                mz, m_invalid = residual_normalise(model_vectors, basis)
                grid = np.einsum("sowp,lwp->solw", ez, mz, optimize=True).astype(np.float32)
                grid[np.broadcast_to(e_invalid[:, :, None, :], grid.shape)] = np.nan
                grid[np.broadcast_to(m_invalid[None, None, :, :], grid.shape)] = np.nan
                group = np.nanmean(grid, axis=0)
                maxima = []
                for oi, scope in enumerate(scopes):
                    layer, window = np.unravel_index(np.nanargmax(group[oi]), group[oi].shape)
                    value = float(group[oi, layer, window])
                    maxima.append(value)
                    rows.append({
                        "dataset": dataset, "model": tag, "emotion": emotion,
                        "control_set": set_name, "retained_controls": "+".join(retained),
                        "scope": scope, "max_partial_rho": value,
                        "max_layer": int(layer), "max_window": int(window),
                    })
                rows.append({
                    "dataset": dataset, "model": tag, "emotion": emotion,
                    "control_set": set_name, "retained_controls": "+".join(retained),
                    "scope": "ROI-avg", "max_partial_rho": float(np.mean(maxima[:6])),
                    "max_layer": -1, "max_window": -1,
                })
                if set_name == "joint":
                    joint_by_emotion.append(grid)

        joint = np.stack(joint_by_emotion, axis=3)  # subject, scope, layer, emotion, window
        np.savez_compressed(
            output_dir / f"{tag}_partial_joint_grid.npz", rsa=joint,
            subjects=np.asarray(subjects), scopes=np.asarray(scopes_out),
            emotions=np.asarray(EMOTIONS), controls=np.asarray(CONTROL_NAMES),
        )
        print(f"[ok] {dataset} {tag}: {joint.shape}", flush=True)

    csv_path = output_dir / "partial_rsa_maxima.csv"
    if args.models and csv_path.exists():
        with csv_path.open(newline="") as handle:
            prior = list(csv.DictReader(handle))
        selected_tags = set(args.models)
        rows = [row for row in prior if row["model"] not in selected_tags] + rows
    fields = ["dataset", "model", "emotion", "control_set", "retained_controls",
              "scope", "max_partial_rho", "max_layer", "max_window"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
