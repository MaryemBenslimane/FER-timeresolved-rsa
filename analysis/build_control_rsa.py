#!/usr/bin/env python3
"""Time-locked RSA for the low-level control RDMs, + the noise-ceiling table.

Controls (identity / pixel_diff / motion_energy / optical_flow) are STATIC 18x18
RDMs - they have no frame dimension - so the same control RDM is correlated with
each of the 16 EEG windows, and max_r is the max over those 16 windows of the
subject-mean Spearman. That is the L=1 case of the model convention in
build_maxr_table.py, so the two are directly comparable.

Note: within a single emotion the 6 stimuli are 6 DIFFERENT actors, so the
identity RDM's 6x6 block is constant and its Spearman is undefined. Those cells
are emitted as NaN rather than 0.

Outputs:
  analysis/motion_controls/results/timelocked_control_rsa.csv
  analysis/paper_stats/noise_ceiling_16win.csv
"""
import csv
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(os.environ.get("FER_ROOT", "."))
EEG_ROI_DIR = ROOT / "eeg_data/Unmasked_avg_windowed_rdms_16win_full"
EEG_WS_DIR = ROOT / "eeg_data/Unmasked_avg_wholescalp_16win"
RDM_DIR = ROOT / "analysis/motion_controls/rdms"
RES_DIR = ROOT / "analysis/motion_controls/results"
NC_OUT = ROOT / "analysis/paper_stats/noise_ceiling_16win.csv"

EMOTIONS = {"all": np.arange(0, 18), "fear": np.arange(0, 6),
            "happy": np.arange(6, 12), "neutral": np.arange(12, 18)}
CONTROLS = ["identity", "identity_graded", "pixel_diff", "motion_energy", "optical_flow"]


def upper(r):
    iu = np.triu_indices(r.shape[-1], 1)
    return r[..., iu[0], iu[1]]


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def load_eeg():
    roi_f = sorted(p for p in EEG_ROI_DIR.glob("*_windowed_rdms.npy")
                   if not p.name.startswith("group"))
    ws_f = sorted(EEG_WS_DIR.glob("*_wholescalp_rdms.npy"))
    roi = np.stack([np.load(p) for p in roi_f], 0)
    ws = np.stack([np.load(p) for p in ws_f], 0)
    rois = [str(x) for x in np.load(EEG_ROI_DIR / "roi_order.npy", allow_pickle=True)]
    return np.concatenate([roi, ws[:, None]], 1), rois + ["wholescalp"]


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    NC_OUT.parent.mkdir(parents=True, exist_ok=True)

    eeg, scopes = load_eeg()
    S, O, W = eeg.shape[0], eeg.shape[1], eeg.shape[2]
    wb_p = EEG_ROI_DIR / "window_bounds.npy"
    wb = np.load(wb_p) if wb_p.exists() else np.zeros((W, 2), int)
    print(f"[eeg] {eeg.shape} scopes={scopes}")

    ctrl_rdms = {}
    for c in CONTROLS:
        p = RDM_DIR / f"{c}_rdm.npy"
        if not p.exists():
            print(f"  [skip] {c}: {p} missing"); continue
        r = np.load(p)
        if r.shape != (18, 18):
            print(f"  [skip] {c}: shape {r.shape} != (18,18)"); continue
        ctrl_rdms[c] = r
    print(f"[controls] {list(ctrl_rdms)}")

    nc_rows, rows = [], []
    for oi, sc in enumerate(scopes):
        for emo, idx in EMOTIONS.items():
            ev = upper(eeg[:, oi][:, :, idx][:, :, :, idx])         # (S, W, P)

            lo = np.full(W, np.nan); up = np.full(W, np.nan)
            for k in range(W):
                gm = ev[:, k].mean(0); tot = ev[:, k].sum(0)
                up[k] = np.nanmean([spearman(ev[s, k], gm) for s in range(S)])
                lo[k] = np.nanmean([spearman(ev[s, k], (tot - ev[s, k]) / (S - 1))
                                    for s in range(S)])
            nc_rows.append({"scope": sc, "roi": sc, "emotion": emo,
                            "lower_mean": round(float(np.nanmean(lo)), 5),
                            "upper_mean": round(float(np.nanmean(up)), 5)})

            for c, rdm in ctrl_rdms.items():
                cv = upper(rdm[np.ix_(idx, idx)])
                per_win = np.array([np.nanmean([spearman(cv, ev[s, k]) for s in range(S)])
                                    for k in range(W)])
                if np.all(np.isnan(per_win)):
                    rows.append({"control": c, "emotion": emo, "roi": sc, "scope": sc,
                                 "max_r": "", "max_window": "", "win_start_ms": "",
                                 "win_end_ms": "", "nc_lower": "", "nc_upper": "",
                                 "within_nc": "undefined"})
                    continue
                ki = int(np.nanargmax(per_win)); v = float(per_win[ki])
                rows.append({
                    "control": c, "emotion": emo, "roi": sc, "scope": sc,
                    "max_r": round(v, 5), "max_window": ki,
                    "win_start_ms": int(wb[ki][0]), "win_end_ms": int(wb[ki][1]),
                    "nc_lower": round(float(lo[ki]), 5), "nc_upper": round(float(up[ki]), 5),
                    "within_nc": "ABOVE" if v > up[ki] else ("below" if v < lo[ki] else "within"),
                })
        print(f"  {sc} done")

    with (RES_DIR / "timelocked_control_rsa.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["control", "emotion", "roi", "scope", "max_r",
                                          "max_window", "win_start_ms", "win_end_ms",
                                          "nc_lower", "nc_upper", "within_nc"])
        w.writeheader(); w.writerows(rows)
    with NC_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "roi", "emotion", "lower_mean", "upper_mean"])
        w.writeheader(); w.writerows(nc_rows)

    n_undef = sum(r["within_nc"] == "undefined" for r in rows)
    print(f"\n[ok] {len(rows)} control rows -> {RES_DIR / 'timelocked_control_rsa.csv'}"
          f"  ({n_undef} undefined)")
    print(f"[ok] {len(nc_rows)} nc rows -> {NC_OUT}")


if __name__ == "__main__":
    main()
