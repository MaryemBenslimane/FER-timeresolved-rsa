"""Central paths, constants, and the model registry for the time-resolved RSA pipeline.

All data locations derive from the FER_ROOT environment variable:

    export FER_ROOT=/path/to/data          # default: /home/maryem/scratch/FER

The canonical 18-stimulus order is EMOTION-MAJOR with numeric actors, matching the
EEG condition order:

    fear    = 0..5      happy = 6..11      neutral = 12..17
    (each block: Actor 01, 02, 04, 07, 08, 11)
"""
import os
from pathlib import Path

import numpy as np

FER_ROOT = Path(os.environ.get("FER_ROOT", "/home/maryem/scratch/FER"))

# --- data ------------------------------------------------------------------
# 16 non-overlapping windows (62-sample, last 70) covering the full 1 s epoch,
# per subject, per ROI:  (n_roi, 16, 18, 18).
EEG_DIR       = FER_ROOT / "eeg_data" / "Unmasked_avg_windowed_rdms_16win_full"
MODEL_RDM_DIR = FER_ROOT / "analysis" / "model_rdms_canonical"
RESULTS_DIR   = FER_ROOT / "analysis" / "paper_stats"

# --- experiment ------------------------------------------------------------
N_SUBJECTS  = 26
N_STIM      = 18
N_WINDOWS   = 16
N_FRAMES    = 16
ROI_ORDER   = ["Ant-Left", "Ant-Right", "Post-Left", "Post-Right", "Temp-Left", "Temp-Right"]

# emotion -> stimulus indices into the 18-stim RDM (canonical emotion-major order)
EMOTIONS = {
    "fear":    np.arange(0, 6),
    "happy":   np.arange(6, 12),
    "neutral": np.arange(12, 18),
    "all":     np.arange(0, 18),
}

# --- model registry (canonical, 16-frame, stimulus-order-corrected RDMs) ----
# label -> tag (file <tag>.npy under MODEL_RDM_DIR, shape (n_layer, 16, 18, 18))
MODELS = {
    "ResNet18 (fine-tuned)":       "resnet18_affectnet_16f",
    "ResNet18 (pretrained)":       "resnet18_affectnet_16f_pretrained",
    "VGGFace (fine-tuned)":        "vggface_affectnet_16f",
    "VGGFace (pretrained)":        "vggface_affectnet_16f_pretrained",
    "CNN2D-CAER (trained)":        "cnn2d_caer_canon",
    "CNN2D-CAER (untrained)":      "cnn2d_caer_random_canon",
    "CNN3D-CAER (trained)":        "cnn3d_caer_canon",
    "CNN3D-CAER (untrained)":      "cnn3d_caer_random_canon",
    "DINOv2-temporal (trained)":   "dinov2_temporal_canon",
    "DINOv2-temporal (untrained)": "dinov2_temporal_random_canon",
    "Qwen2.5-VL":                  "qwen_vl_video_mode_canon",
}

# trained/untrained (or fine-tuned/pretrained) pairs for direct comparison
MODEL_PAIRS = [
    ("ResNet18-AffectNet", ("fine-tuned", "resnet18_affectnet_16f"),
                           ("pretrained (ImageNet)", "resnet18_affectnet_16f_pretrained")),
    ("VGGFace-AffectNet",  ("fine-tuned", "vggface_affectnet_16f"),
                           ("pretrained (VGGFace2)", "vggface_affectnet_16f_pretrained")),
    ("CNN2D-CAER",         ("trained", "cnn2d_caer_canon"),
                           ("untrained", "cnn2d_caer_random_canon")),
    ("CNN3D-CAER",         ("trained", "cnn3d_caer_canon"),
                           ("untrained", "cnn3d_caer_random_canon")),
    ("DINOv2-temporal",    ("trained", "dinov2_temporal_canon"),
                           ("untrained", "dinov2_temporal_random_canon")),
    ("Qwen2.5-VL",         ("video", "qwen_vl_video_mode_canon"), None),
]

# --- plotting: ROI colour (region) x line style (hemisphere), CVD-safe ------
ROI_STYLE = {
    "Ant-Left":  ("#0072B2", "-"), "Ant-Right":  ("#0072B2", "--"),
    "Post-Left": ("#009E73", "-"), "Post-Right": ("#009E73", "--"),
    "Temp-Left": ("#D55E00", "-"), "Temp-Right": ("#D55E00", "--"),
}

# --- inference -------------------------------------------------------------
N_BOOT = 10_000
N_PERM = 10_000
SEED   = 20260726


def window_centers():
    """Centre (ms) of each of the 16 windows; the last window is wider (930-1000)."""
    c = np.array([62 * k + 31 for k in range(N_WINDOWS)], dtype=float)
    c[-1] = (930 + 1000) / 2.0
    return c
