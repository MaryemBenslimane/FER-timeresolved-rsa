"""Time-locked RSA time courses: one panel per model, all 6 ROIs overlaid.

    python figures/plot_timecourses.py                          # max 1 layer, all stimuli
    python figures/plot_timecourses.py --selection pertimepoint # max over all layers
    python figures/plot_timecourses.py --emotion fear           # within one emotion

--selection:
    fixed        one LOSO-selected layer, held across the epoch   ("max 1 layer")
    pertimepoint layer re-chosen at each timepoint (LOSO)         ("max over all layers")
    NB the two are not on a comparable scale (16 selections vs 1 inflates 'pertimepoint').

Output: figures written to RESULTS_DIR as rsa_timecourses[_pertimepoint][_<emotion>].{pdf,png,npz}
"""
import argparse
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
from fer_tr.core import load_eeg_per_roi, load_model_rdm, timecourse, timecourse_pertimepoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", choices=["fixed", "pertimepoint"], default="fixed")
    ap.add_argument("--emotion", choices=list(config.EMOTIONS), default="all")
    args = ap.parse_args()
    tag_sel = "" if args.selection == "fixed" else "_pertimepoint"
    tag_emo = "" if args.emotion == "all" else f"_{args.emotion}"
    title_sel = "max 1 layer" if args.selection == "fixed" else "max over all layers"

    eeg, rois = load_eeg_per_roi(args.emotion)
    centers = config.window_centers()
    present = [(lab, tag) for lab, tag in config.MODELS.items()
               if (Path(config.MODEL_RDM_DIR) / f"{tag}.npy").exists()]

    ncols = 4
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.6 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    store = {}
    for ax, (label, tag) in zip(axes, present):
        model = load_model_rdm(tag, args.emotion)
        for ri, roi in enumerate(rois):
            if args.selection == "fixed":
                mean, sem = timecourse(eeg[:, ri], model)
            else:
                mean, sem, _ = timecourse_pertimepoint(eeg[:, ri], model)
            if mean is None:
                continue
            color, ls = config.ROI_STYLE.get(roi, ("#666666", "-"))
            ax.plot(centers, mean, ls, color=color, lw=1.5, label=roi)
            ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.13, lw=0)
            store[f"{tag}|{roi}"] = np.vstack([mean, sem])
        ax.axhline(0, color="#999999", lw=0.9, ls=":")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes[len(present):]:
        ax.set_visible(False)
    for ax in axes[:len(present)]:
        ax.set_xlabel("EEG window centre (ms)", fontsize=8)
    axes[0].set_ylabel("time-locked RSA (Spearman)", fontsize=8)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    emo_txt = "" if args.emotion == "all" else f" — {args.emotion}"
    fig.suptitle(f"Time-locked model–EEG RSA{emo_txt} — {title_sel}, per ROI",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    out = Path(config.RESULTS_DIR); out.mkdir(parents=True, exist_ok=True)
    stem = f"rsa_timecourses{tag_sel}{tag_emo}"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", dpi=200)
        print(f"[ok] -> {out / f'{stem}.{ext}'}")
    np.savez(out / f"{stem}.npz", centers=centers, **store)


if __name__ == "__main__":
    main()
