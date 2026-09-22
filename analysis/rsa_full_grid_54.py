#!/usr/bin/env python3
"""Time-locked EEG/model RSA for the 54-stimulus occlusion study."""
import argparse
import csv
from pathlib import Path

import numpy as np

from rsa_full_grid import upper, rank_normalise

ROOT = Path(".")
EEG = ROOT / "eeg_rdms_complete"
MDIR = ROOT / "analysis/model_rdms_54"
EMOTIONS = ("all", "fear", "happy", "neutral")


def subjects():
    roi = {p.name.split("_", 1)[0] for p in (EEG / "perroi").glob("*_all_rdms.npy")}
    ws = {p.name.split("_", 1)[0] for p in (EEG / "wholescalp").glob("*_all_rdms.npy")}
    if roi != ws:
        raise RuntimeError(f"subject mismatch: ROI-only={roi-ws}, WS-only={ws-roi}")
    return sorted(roi)


def load_eeg(subs, emotion):
    roi = np.stack([np.load(EEG / "perroi" / f"{s}_{emotion}_rdms.npy") for s in subs])
    ws = np.stack([np.load(EEG / "wholescalp" / f"{s}_{emotion}_rdms.npy") for s in subs])
    return np.concatenate([roi, ws[:, None]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "analysis/rsa_grid_54"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    subs = subjects()
    rois = [str(x) for x in np.load(EEG / "roi_order.npy", allow_pickle=True)]
    scopes = rois + ["wholescalp"]
    eeg = {emotion: load_eeg(subs, emotion) for emotion in EMOTIONS}
    W = eeg["all"].shape[2]
    bounds = np.load(EEG / "window_bounds.npy")
    summary = []

    for tag in args.models:
        model_all = np.load(MDIR / f"{tag}.npy")
        model = {"all": model_all}
        model.update({emotion: np.load(MDIR / f"{tag}_{emotion}.npy")
                      for emotion in EMOTIONS[1:]})
        L = model_all.shape[0]
        grid = np.full((len(subs), 7, L, 4, W), np.nan, np.float32)
        for ei, emotion in enumerate(EMOTIONS):
            erdm, mrdm = eeg[emotion], model[emotion]
            if erdm.shape[-1] != mrdm.shape[-1]:
                raise RuntimeError(f"{tag}/{emotion}: EEG {erdm.shape}, model {mrdm.shape}")
            ev, mv = upper(erdm), upper(mrdm[:, :W])
            ez, eflat = rank_normalise(ev)
            mz, mflat = rank_normalise(mv)
            rsa = np.einsum("sowp,lwp->solw", ez, mz, optimize=True)
            rsa[np.broadcast_to(mflat[None, None], rsa.shape)] = np.nan
            rsa[np.broadcast_to(eflat[:, :, None], rsa.shape)] = np.nan
            grid[:, :, :, ei] = rsa.astype(np.float32)

        np.savez_compressed(out_dir / f"{tag}_rsa_grid.npz", rsa=grid,
                            subjects=np.array(subs), scopes=np.array(scopes),
                            emotions=np.array(EMOTIONS), n_layers=L,
                            window_bounds=bounds)
        tl = np.nanmean(grid, axis=-1)
        with (out_dir / f"{tag}_rsa.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "subject", "scope", "layer", "emotion", "rsa_timelocked"])
            for si, sub in enumerate(subs):
                for oi, scope in enumerate(scopes):
                    for li in range(L):
                        for ei, emotion in enumerate(EMOTIONS):
                            value = tl[si, oi, li, ei]
                            w.writerow([tag, sub, scope, li, emotion, f"{value:.6f}"])
        best = float(np.nanmax(np.nanmean(tl, axis=0)))
        print(f"[ok] {tag}: model={model_all.shape}, grid={grid.shape}, best={best:+.4f}")


if __name__ == "__main__":
    main()
