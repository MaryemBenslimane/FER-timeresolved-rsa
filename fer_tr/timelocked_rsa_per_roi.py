"""General time-locked RSA: any model RDM(s) vs the 16-window EEG, per-ROI + whole-scalp.

Consumes model RDMs produced by models/extract_model_rdms.py (shape (n_layer,16,18,18),
correlation-distance, canonical emotion-major stimulus order) and correlates them with
the time-locked 16-window EEG RDMs, using window k <-> model-frame k (the diagonal).

For every model x scope x condition it reports:
  max_r  : max over (layer x 16 windows) of the group-mean time-locked Spearman
           (window k <-> model-frame k), with the winning layer index, window interval,
           and a noise-ceiling verdict at that window (within / below lower / ABOVE upper).

Scopes : the 6 scalp ROIs (Ant-Left ... Temp-Right)  +  "wholescalp" (all 42 channels).
Conditions : all (18 stim) / fear (0-5) / happy (6-11) / neutral (12-17).

EEG per-ROI    : eeg_data/Unmasked_avg_windowed_rdms_16win_full/*_windowed_rdms.npy  (S,6,16,18,18)
EEG wholescalp : eeg_data/Unmasked_avg_wholescalp_16win/*_wholescalp_rdms.npy         (S,16,18,18)
Noise ceiling  : computed here per (scope, condition, window).

Usage:
  python timelocked_rsa_general.py --models facenet_withmeg_pretrained_16f vgg16bn_nomeg_16f
  python timelocked_rsa_general.py --models <tag> --out my_results.csv
  (tags = basenames under analysis/model_rdms_canonical/, without .npy)

Output: analysis/paper_stats/<out>.csv  (default timelocked_rsa_general.csv)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(".")
EEG_ROI_DIR = ROOT / "eeg_data/Unmasked_avg_windowed_rdms_16win_full"
EEG_WS_DIR = ROOT / "eeg_data/Unmasked_avg_wholescalp_16win"
MDIR = ROOT / "analysis/model_rdms_canonical"
OUT = ROOT / "analysis/paper_stats"
WINB = np.load(EEG_ROI_DIR / "window_bounds.npy")          # (16,2) ms
EMO = {"all": list(range(18)), "fear": list(range(0, 6)),
       "happy": list(range(6, 12)), "neutral": list(range(12, 18))}


# ---- rank / RSA helpers ----
def upper(r):
    n = r.shape[-1]; iu = np.triu_indices(n, 1); return r[..., iu[0], iu[1]]
def rn(v):
    r = np.apply_along_axis(rankdata, -1, v); rc = r - r.mean(-1, keepdims=True)
    return np.divide(rc, np.clip(np.linalg.norm(rc, axis=-1, keepdims=True), 1e-12, None))
def spear(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return np.nan if np.std(rx) == 0 or np.std(ry) == 0 else float(np.corrcoef(rx, ry)[0, 1])


# ---- EEG loaders (subject stacks) ----
def load_roi_eeg():
    fs = sorted(p for p in EEG_ROI_DIR.glob("*_windowed_rdms.npy") if not p.name.startswith("group"))
    stack = np.stack([np.load(p) for p in fs], 0)          # (S,6,16,18,18)
    rois = [str(x) for x in np.load(EEG_ROI_DIR / "roi_order.npy", allow_pickle=True)]
    return stack, rois
def load_ws_eeg():
    fs = sorted(EEG_WS_DIR.glob("*_wholescalp_rdms.npy"))
    return np.stack([np.load(p) for p in fs], 0)           # (S,16,18,18)


# ---- time-locked estimators (operate on flattened upper-tri vectors) ----
def noise_ceiling(ev):
    """ev (S,16,P) -> per-window (lower, upper)."""
    S, W, _ = ev.shape
    lo = np.full((S, W), np.nan); up = np.full((S, W), np.nan)
    for k in range(W):
        gm = ev[:, k].mean(0)
        for s in range(S):
            up[s, k] = spear(ev[s, k], gm)
            up_loo = (ev[:, k].sum(0) - ev[s, k]) / (S - 1)
            lo[s, k] = spear(ev[s, k], up_loo)
    return np.nanmean(lo, 0), np.nanmean(up, 0)

def maxr(ev, mv):
    """max over (layer x window) of group-mean r -> (max_r, layer, window)."""
    S, W = ev.shape[0], ev.shape[1]
    rg = np.einsum("skp,lkp->lk", rn(ev), rn(mv)) / S
    deg = np.stack([[np.std(mv[l, k]) == 0 for k in range(W)] for l in range(mv.shape[0])])
    rg[deg] = np.nan
    if np.all(np.isnan(rg)):
        return None
    l, k = np.unravel_index(int(np.nanargmax(rg)), rg.shape)
    return float(rg[l, k]), int(l), int(k)


def score(ev_full, model_full, scope, model_tag, nc_cache):
    """ev_full (S,16,18,18), model_full (L,16,18,18). Emit one row per condition."""
    rows = []
    for cond, idx in EMO.items():
        ev = upper(ev_full[:, :, idx][:, :, :, idx])       # (S,16,P)
        mv = upper(model_full[:, :, idx][:, :, :, idx])    # (L,16,P)
        key = (scope, cond)
        if key not in nc_cache:
            nc_cache[key] = noise_ceiling(ev)
        nclo, ncup = nc_cache[key]
        mr = maxr(ev, mv)
        if mr is None:
            continue
        mval, li, ki = mr
        verdict = "ABOVE" if mval > ncup[ki] else ("below" if mval < nclo[ki] else "within")
        rows.append({"model": model_tag, "scope": scope, "condition": cond, "n_layers": model_full.shape[0],
                     "max_r": round(mval, 4), "max_layer": li,
                     "win_start_ms": int(WINB[ki][0]), "win_end_ms": int(WINB[ki][1]),
                     "nc_lower": round(float(nclo[ki]), 4), "nc_upper": round(float(ncup[ki]), 4),
                     "within_nc": verdict})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="model RDM basenames (no .npy) under model_rdms_canonical/")
    ap.add_argument("--out", default="timelocked_rsa_general.csv", help="output CSV name")
    ap.add_argument("--scopes", nargs="+", default=["roi", "wholescalp"], choices=["roi", "wholescalp"])
    args = ap.parse_args()

    eeg_roi, rois = (load_roi_eeg() if "roi" in args.scopes else (None, []))
    eeg_ws = load_ws_eeg() if "wholescalp" in args.scopes else None
    print(f"ROIs: {rois}  | wholescalp: {None if eeg_ws is None else eeg_ws.shape}")

    nc_cache = {}; rows = []
    for tag in args.models:
        fp = MDIR / f"{tag}.npy"
        if not fp.exists():
            print(f"[skip] {tag}.npy not found"); continue
        M = np.load(fp)                                    # (L,16,18,18)
        if M.ndim != 4 or M.shape[1] != 16 or M.shape[2:] != (18, 18):
            print(f"[skip] {tag}: unexpected shape {M.shape}"); continue
        print(f"\n#### {tag}  (L={M.shape[0]}) ####")
        if eeg_roi is not None:
            for ri, roi in enumerate(rois):
                rows += score(eeg_roi[:, ri], M, roi, tag, nc_cache)
        if eeg_ws is not None:
            rows += score(eeg_ws, M, "wholescalp", tag, nc_cache)
        # print a compact per-model summary
        for r in [x for x in rows if x["model"] == tag]:
            print(f"  {r['scope']:<11}{r['condition']:<8} "
                  f"max_r={r['max_r']:+.3f} L{r['max_layer']} {r['win_start_ms']}-{r['win_end_ms']}ms {r['within_nc']}")

    if not rows:
        print("no rows produced"); return
    outp = OUT / args.out
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"\n[ok] {len(rows)} rows -> {outp}")


if __name__ == "__main__":
    main()
