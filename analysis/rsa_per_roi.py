"""Time-locked, per-ROI RSA for every model. Optionally within each emotion.

    python analysis/rsa_per_roi.py                 # all-stimulus  -> rsa_per_roi.csv
    python analysis/rsa_per_roi.py --per-emotion   # per emotion   -> rsa_per_roi_per_emotion.csv

Estimator: fer_tr.core.timelocked_rsa (EEG window k vs model frame k, ONE LOSO-selected). 
Reports subject-mean RSA, 95% bootstrap CI and sign-flip permutation p; 
per-emotion runs also add Benjamini-Hochberg FDR within each emotion family (models x ROIs).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from fer_tr.core import (load_eeg_per_roi, load_model_rdm, timelocked_rsa,
                         bootstrap_ci, signflip_p, bh_fdr)


def run(emotions):
    rows = []
    for emo in emotions:
        eeg, rois = load_eeg_per_roi(emo)
        n_pairs = len(config.EMOTIONS[emo]) * (len(config.EMOTIONS[emo]) - 1) // 2
        print(f"\n#### {emo} ({n_pairs} pairs)  EEG {eeg.shape} ####")
        for label, tag in config.MODELS.items():
            try:
                model = load_model_rdm(tag, emo)
            except FileNotFoundError:
                print(f"[skip] {tag}: missing"); continue
            print(f"=== {label} ===")
            for ri, roi in enumerate(rois):
                tl = timelocked_rsa(eeg[:, ri], model)
                if tl is None:
                    print(f"  {roi:<12s} skipped (degenerate)"); continue
                lo, hi = bootstrap_ci(tl); p = signflip_p(tl)
                rows.append({"emotion": emo, "model": tag, "label": label, "roi": roi,
                             "n": len(tl), "n_pairs": n_pairs,
                             "timelock_rsa": float(tl.mean()),
                             "ci_lo": lo, "ci_hi": hi, "p_perm": p})
                print(f"  {roi:<12s} rsa={tl.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")

    # BH-FDR within each emotion family
    for emo in emotions:
        sub = [r for r in rows if r["emotion"] == emo]
        if not sub:
            continue
        q = bh_fdr([r["p_perm"] for r in sub])
        for r, qq in zip(sub, q):
            r["q_fdr"] = float(qq); r["sig_fdr"] = bool(qq < 0.05)
        print(f"[{emo}] {sum(r['sig_fdr'] for r in sub)}/{len(sub)} survive FDR q<0.05")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-emotion", action="store_true",
                    help="analyse fear/happy/neutral/all separately (default: all only)")
    args = ap.parse_args()
    emotions = ["fear", "happy", "neutral", "all"] if args.per_emotion else ["all"]
    stem = "rsa_per_roi_per_emotion" if args.per_emotion else "rsa_per_roi"

    rows = run(emotions)
    out = Path(config.RESULTS_DIR); out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(rows, indent=2))
    cols = ["emotion", "model", "label", "roi", "n", "n_pairs", "timelock_rsa",
            "ci_lo", "ci_hi", "p_perm", "q_fdr", "sig_fdr"]
    with (out / f"{stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k))
                        for k in cols})
    print(f"\n[ok] -> {out / f'{stem}.csv'}")


if __name__ == "__main__":
    main()
