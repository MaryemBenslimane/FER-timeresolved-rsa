"""Shared RSA / statistics primitives for the time-resolved pipeline.

Every analysis and figure script imports from here, so the estimator is defined
exactly once. The design decisions baked in:

* Time-locked correspondence -- EEG window k is only ever compared with model frame
  k (the 16 axes are matched 1:1). Time is never searched.
* 6 bilateral ROIs are analysed independently.
* Layer selection is leave-one-subject-out (LOSO): the layer is chosen from the other
  25 subjects and the held-out subject is scored there, so it is never circular.
    - `timelocked_rsa` / `timecourse` : ONE layer, chosen by its average diagonal score
      over the whole epoch, held fixed across the 16 timepoints  ("max 1 layer").
    - `timecourse_pertimepoint`       : the layer is re-chosen at EACH timepoint
      ("max over all layers"). This has 16 selections instead of 1
* Degenerate (constant) model RDMs give zero-variance vectors and NaN correlations;
  such layers/timepoints are excluded from selection and averaging.
"""
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import config


def _rng_for(values):
    """A per-call RNG seeded deterministically from the input vector, so bootstrap CIs
    and permutation p-values are reproducible and INDEPENDENT of how many other tests
    ran before (a module-level RNG consumed in sequence is not)."""
    v = np.asarray(values, float)
    h = np.uint64(config.SEED)
    for x in np.round(v, 9):
        h = (h * np.uint64(1099511628211) + np.uint64(int(x * 1e9) & 0xFFFFFFFF)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return np.random.default_rng(int(h))


# --------------------------------------------------------------------------- #
# RDM vectorisation                                                           #
# --------------------------------------------------------------------------- #
def upper(rdm):
    """(..., n, n) -> (..., n*(n-1)/2) upper-triangle vector."""
    n = rdm.shape[-1]
    iu = np.triu_indices(n, 1)
    return rdm[..., iu[0], iu[1]]


def rank_normalize(v):
    """Rank, mean-centre and L2-normalise the last axis, so a dot product == Spearman."""
    r = np.apply_along_axis(rankdata, -1, v)
    rc = r - r.mean(-1, keepdims=True)
    return np.divide(rc, np.clip(np.linalg.norm(rc, axis=-1, keepdims=True), 1e-12, None))


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def slice_emotion(rdm, idx):
    """Sub-RDM for a set of stimulus indices (trailing two axes)."""
    idx = np.asarray(idx)
    return rdm[..., idx[:, None], idx[None, :]]


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_eeg_per_roi(emotion="all"):
    """Stack per-subject windowed RDMs -> (S, n_roi, W, k, k); slice to an emotion.

    NO ROI averaging. Returns (eeg, roi_names).
    """
    fs = sorted(p for p in Path(config.EEG_DIR).glob("*_windowed_rdms.npy")
                if not p.name.startswith("group"))
    if not fs:
        raise FileNotFoundError(f"No *_windowed_rdms.npy in {config.EEG_DIR}")
    stack = np.stack([np.load(p) for p in fs], 0)          # (S, n_roi, W, 18, 18)
    if stack.ndim != 5:
        raise ValueError(f"Expected (S, n_roi, W, 18, 18), got {stack.shape}")
    roi_path = Path(config.EEG_DIR) / "roi_order.npy"
    rois = ([str(x) for x in np.load(roi_path, allow_pickle=True)]
            if roi_path.exists() else [f"roi{i}" for i in range(stack.shape[1])])
    return slice_emotion(stack, config.EMOTIONS[emotion]), rois


def load_model_rdm(tag, emotion="all"):
    """(n_layer, W, k, k) model RDM, sliced to an emotion. Raises if missing."""
    p = Path(config.MODEL_RDM_DIR) / f"{tag}.npy"
    if not p.exists():
        raise FileNotFoundError(p)
    return slice_emotion(np.load(p), config.EMOTIONS[emotion])


def _usable_layers(mvec, W):
    """Layers whose RDM has non-zero variance at EVERY timepoint (else NaN Spearman)."""
    return np.array([all(np.std(mvec[l, k]) > 0 for k in range(W))
                     for l in range(mvec.shape[0])])


# --------------------------------------------------------------------------- #
# Estimators                                                                  #
# --------------------------------------------------------------------------- #
def timelocked_rsa(eeg_roi, model):
    """Per-subject time-locked RSA, ONE LOSO-selected layer fixed across the epoch.

    eeg_roi (S, W, k, k), model (L, W, k, k) -> length-S vector (NaNs dropped) or None.
    """
    n, W = eeg_roi.shape[0], eeg_roi.shape[1]
    if W != model.shape[1]:
        return None
    ev = upper(eeg_roi); ez = rank_normalize(ev)
    mvec = upper(model); mz = rank_normalize(mvec)
    usable = _usable_layers(mvec, W)
    if not usable.any():
        return None
    diag = np.einsum("swp,lwp->sl", ez, mz) / W            # average over timepoints
    out = np.full(n, np.nan)
    for s in range(n):
        others = np.delete(np.arange(n), s)
        gm = diag[others].mean(0).copy(); gm[~usable] = -np.inf
        l = int(np.nanargmax(gm))
        out[s] = float(np.nanmean([spearman(mvec[l, k], ev[s, k]) for k in range(W)]))
    out = out[~np.isnan(out)]
    return out if len(out) else None


def timecourse(eeg_roi, model):
    """Fixed-layer ('max 1 layer') RSA per timepoint -> (mean[W], sem[W]) or (None, None)."""
    n, W = eeg_roi.shape[0], eeg_roi.shape[1]
    ev = upper(eeg_roi); ez = rank_normalize(ev)
    mvec = upper(model); mz = rank_normalize(mvec)
    usable = _usable_layers(mvec, W)
    if not usable.any():
        return None, None
    diag = np.einsum("swp,lwp->sl", ez, mz) / W
    curves = np.full((n, W), np.nan)
    for s in range(n):
        others = np.delete(np.arange(n), s)
        gm = diag[others].mean(0).copy(); gm[~usable] = -np.inf
        l = int(np.nanargmax(gm))
        curves[s] = [spearman(mvec[l, k], ev[s, k]) for k in range(W)]
    return _mean_sem(curves)


def timecourse_pertimepoint(eeg_roi, model):
    """Per-timepoint-argmax ('max over all layers') RSA -> (mean[W], sem[W], modal[W])."""
    n, W = eeg_roi.shape[0], eeg_roi.shape[1]
    ev = upper(eeg_roi); ez = rank_normalize(ev)
    mvec = upper(model); mz = rank_normalize(mvec)
    L = mvec.shape[0]
    ok = np.array([[np.std(mvec[l, k]) > 0 for k in range(W)] for l in range(L)])
    if not ok.any():
        return None, None, None
    R = np.einsum("swp,lwp->slw", ez, mz)                  # (S, L, W) Spearman
    curves = np.full((n, W), np.nan)
    chosen = np.zeros((n, W), dtype=int)
    for s in range(n):
        others = np.delete(np.arange(n), s)
        gm = np.where(ok, R[others].mean(0), -np.inf)
        for k in range(W):
            if not np.isfinite(gm[:, k]).any():
                continue
            l = int(np.argmax(gm[:, k]))
            chosen[s, k] = l
            curves[s, k] = R[s, l, k]
    mean, sem = _mean_sem(curves)
    modal = np.array([np.bincount(chosen[:, k], minlength=L).argmax() for k in range(W)])
    return mean, sem, modal


def _mean_sem(curves):
    mean = np.nanmean(curves, axis=0)
    sem = np.nanstd(curves, axis=0) / np.sqrt(np.sum(~np.isnan(curves), axis=0))
    return mean, sem


# --------------------------------------------------------------------------- #
# Noise ceiling (Nili et al., 2014)                                           #
# --------------------------------------------------------------------------- #
def noise_ceiling(eeg_roi):
    """eeg_roi (S, W, k, k) -> (upper[W], lower[W], sem_upper[W]).

    upper: subject vs group mean INCLUDING itself (overestimate).
    lower: subject vs group mean of the OTHER subjects (leave-one-out, underestimate).
    """
    ev = upper(eeg_roi)                                     # (S, W, P)
    S, W, _ = ev.shape
    up = np.full((S, W), np.nan)
    lo = np.full((S, W), np.nan)
    for k in range(W):
        gm_all = ev[:, k].mean(0)
        tot = ev[:, k].sum(0)
        for s in range(S):
            up[s, k] = spearman(ev[s, k], gm_all)
            lo[s, k] = spearman(ev[s, k], (tot - ev[s, k]) / (S - 1))
    return (np.nanmean(up, 0), np.nanmean(lo, 0),
            np.nanstd(up, 0) / np.sqrt(np.sum(~np.isnan(up), 0)))


# --------------------------------------------------------------------------- #
# Inference                                                                    #
# --------------------------------------------------------------------------- #
def bootstrap_ci(values, n_boot=None):
    v = np.asarray(values); n_boot = n_boot or config.N_BOOT
    rng = _rng_for(v)
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def signflip_p(values, n_perm=None):
    v = np.asarray(values); n_perm = n_perm or config.N_PERM
    obs = abs(v.mean())
    s = _rng_for(v).choice([-1.0, 1.0], size=(n_perm, len(v)))
    return float(((np.abs((s * v).mean(1)) >= obs).sum() + 1) / (n_perm + 1))


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values (monotone-enforced)."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); q = np.empty(m); prev = 1.0
    for rank, i in enumerate(order[::-1]):
        prev = min(prev, p[i] * m / (m - rank))
        q[i] = prev
    return q
