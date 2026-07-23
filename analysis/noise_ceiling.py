"""Noise ceiling of the 16-window EEG RDMs (Nili et al., 2014), per ROI, per timepoint.

    python analysis/noise_ceiling.py    -> noise_ceiling_16win.{csv,json,npz}

Upper bound: subject vs group mean including itself (overestimate; the ceiling).
Lower bound: subject vs leave-one-out group mean (underestimate).
Computed per ROI and within each emotion
comparable to the time-locked RSA in rsa_per_roi*.csv. The .npz stores the full
per-timepoint bounds ("<emotion>|<roi>|{upper,lower}") for the overlay figure.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from fer_tr.core import load_eeg_per_roi, noise_ceiling


def main():
    out = Path(config.RESULTS_DIR); out.mkdir(parents=True, exist_ok=True)
    centers = config.window_centers()
    rows, store = [], {}
    for emo in ["fear", "happy", "neutral", "all"]:
        eeg, rois = load_eeg_per_roi(emo)
        n_pairs = len(config.EMOTIONS[emo]) * (len(config.EMOTIONS[emo]) - 1) // 2
        print(f"\n=== {emo} ({n_pairs} pairs)  EEG {eeg.shape} ===")
        for ri, roi in enumerate(rois):
            up, lo, _sem = noise_ceiling(eeg[:, ri])
            store[f"{emo}|{roi}|upper"] = up
            store[f"{emo}|{roi}|lower"] = lo
            rows.append({"emotion": emo, "roi": roi, "n_pairs": n_pairs,
                         "upper_mean": float(np.nanmean(up)),
                         "upper_min": float(np.nanmin(up)), "upper_max": float(np.nanmax(up)),
                         "lower_mean": float(np.nanmean(lo))})
            print(f"  {roi:<12s} upper={np.nanmean(up):.4f} "
                  f"({np.nanmin(up):.3f}-{np.nanmax(up):.3f})  lower={np.nanmean(lo):.4f}")

    (out / "noise_ceiling_16win.json").write_text(json.dumps(rows, indent=2))
    cols = ["emotion", "roi", "n_pairs", "upper_mean", "upper_min", "upper_max", "lower_mean"]
    with (out / "noise_ceiling_16win.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in cols})
    np.savez(out / "noise_ceiling_16win.npz", centers=centers, **store)
    print(f"\n[ok] -> {out / 'noise_ceiling_16win.csv'}  (+ .npz for the overlay figure)")


if __name__ == "__main__":
    main()
