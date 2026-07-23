"""One heatmap per emotion: model (rows) x ROI (cols) of the time-locked RSA value.

    python figures/plot_emotion_heatmaps.py     (reads rsa_per_roi_per_emotion.csv)

Diverging colour around zero on a shared scale across emotions; the raw value is
printed in each cell. Significance (p_perm, q_fdr) lives in the CSV, not the figure.
Output: rsa_heatmap_<emotion>.{pdf,png}
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

OUT = Path(config.RESULTS_DIR)
CSV = OUT / "rsa_per_roi_per_emotion.csv"
SHORT = {t: (lab.replace("-CAER", "").replace("-temporal", "")
             .replace(" (fine-tuned)", " (ft)").replace(" (pretrained)", " (pre)"))
         for lab, t in config.MODELS.items()}


def main():
    if not CSV.exists():
        raise SystemExit(f"missing {CSV} -- run: python analysis/rsa_per_roi.py --per-emotion")
    rows = list(csv.DictReader(open(CSV)))
    models = [config.MODELS[l] for l in config.MODELS if config.MODELS[l] in {r["model"] for r in rows}]
    emotions = [e for e in config.EMOTIONS if any(r["emotion"] == e for r in rows)]
    rois = [r for r in config.ROI_ORDER if any(x["roi"] == r for x in rows)]
    lut = {(r["emotion"], r["model"], r["roi"]): r for r in rows}
    vmax = max(abs(float(r["timelock_rsa"])) for r in rows)
    print(f"{len(rows)} cells; colour scale +/- {vmax:.3f}")

    for emo in emotions:
        M = np.full((len(models), len(rois)), np.nan)
        for i, m in enumerate(models):
            for j, roi in enumerate(rois):
                rec = lut.get((emo, m, roi))
                if rec:
                    M[i, j] = float(rec["timelock_rsa"])
        fig, ax = plt.subplots(figsize=(4.6, 0.42 * len(models) + 2.0))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(models)):
            for j in range(len(rois)):
                if np.isnan(M[i, j]):
                    continue
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "#222222")
        ax.set_title(f"Time-locked RSA — {emo}", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(rois))); ax.set_xticklabels(rois, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(models))); ax.set_yticklabels([SHORT[m] for m in models], fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(rois), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4); ax.tick_params(which="minor", length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cb.set_label("time-locked RSA (Spearman)", fontsize=8); cb.ax.tick_params(labelsize=7)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"rsa_heatmap_{emo}.{ext}", bbox_inches="tight", dpi=200)
            print(f"[ok] -> {OUT / f'rsa_heatmap_{emo}.{ext}'}")
        plt.close(fig)


if __name__ == "__main__":
    main()
