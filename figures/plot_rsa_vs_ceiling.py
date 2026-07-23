"""Per-model RSA vs. noise ceiling, trained & untrained overlaid, one subplot per ROI.

    python figures/plot_rsa_vs_ceiling.py     (needs noise_ceiling_16win.npz)

For each trained/untrained (or fine-tuned/pretrained) pair in config.MODEL_PAIRS, a
figure with 6 ROI subplots: both time courses (fixed-layer, all-stimulus) plus the
grey noise-ceiling band [lower, upper]. A curve near the LOWER bound is doing as well
as the reliable EEG variance allows; a curve near zero is not.

Output: rsa_vs_ceiling_<name>.{pdf,png}
"""
import os
import re
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
from fer_tr.core import load_eeg_per_roi, load_model_rdm, timecourse

TRAINED, UNTRAINED, CEIL = "#0072B2", "#D55E00", "#888888"


def main():
    nc_path = Path(config.RESULTS_DIR) / "noise_ceiling_16win.npz"
    if not nc_path.exists():
        raise SystemExit("missing noise_ceiling_16win.npz -- run: python analysis/noise_ceiling.py")
    NC = np.load(nc_path)
    centers = config.window_centers()
    eeg, rois = load_eeg_per_roi("all")

    # cache one time course per model tag
    curves = {}
    for label, tag in config.MODELS.items():
        if not (Path(config.MODEL_RDM_DIR) / f"{tag}.npy").exists():
            continue
        model = load_model_rdm(tag, "all")
        curves[tag] = {roi: timecourse(eeg[:, ri], model) for ri, roi in enumerate(rois)}

    for pair in config.MODEL_PAIRS:
        name = pair[0]
        series = [s for s in pair[1:] if s is not None]           # [(label, tag), ...]
        if not all(tag in curves for _, tag in series):
            print(f"[skip] {name}: missing model(s)"); continue
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, roi in zip(axes, config.ROI_ORDER):
            up, lo = NC[f"all|{roi}|upper"], NC[f"all|{roi}|lower"]
            ax.fill_between(centers, lo, up, color=CEIL, alpha=0.18, lw=0, zorder=0,
                            label="noise ceiling")
            ax.plot(centers, up, color=CEIL, lw=1.0, zorder=1)
            ax.plot(centers, lo, color=CEIL, lw=1.0, ls="--", zorder=1)
            for (lab, tag), col in zip(series, (TRAINED, UNTRAINED)):
                mean, sem = curves[tag][roi]
                if mean is None:
                    continue
                ax.plot(centers, mean, color=col, lw=1.7, label=lab, zorder=3)
                ax.fill_between(centers, mean - sem, mean + sem, color=col, alpha=0.15, lw=0, zorder=2)
            ax.axhline(0, color="#999999", lw=0.8, ls=":", zorder=1)
            ax.set_title(roi, fontsize=9, fontweight="bold")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
        for ax in axes[3:]:
            ax.set_xlabel("EEG window centre (ms)", fontsize=8)
        axes[0].set_ylabel("Spearman"); axes[3].set_ylabel("Spearman")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"{name}: time-locked RSA vs. EEG noise ceiling (all stimuli), per ROI",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        for ext in ("pdf", "png"):
            f = Path(config.RESULTS_DIR) / f"rsa_vs_ceiling_{slug}.{ext}"
            fig.savefig(f, bbox_inches="tight", dpi=200)
            print(f"[ok] -> {f}")
        plt.close(fig)


if __name__ == "__main__":
    main()
