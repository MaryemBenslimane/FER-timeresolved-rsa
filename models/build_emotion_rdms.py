#!/usr/bin/env python3
"""Split every canonical (L,16,18,18) RDM into one RDM per emotion: (L,16,6,6).

Slicing the 18x18 is exactly equivalent to recomputing the RDM on that emotion's
6 stimuli alone: corr_distance_rdm centres each stimulus's feature vector by its
own mean, so entry (i,j) depends only on rows i and j, never on the other
stimuli. `--self-test` proves this numerically before anything is written.

Outputs, per input <base>.npy:
    <base>_fear.npy  <base>_happy.npy  <base>_neutral.npy      (L,16,6,6)
    <base>_byemotion.json                                       provenance
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

EMOTIONS = {"fear": slice(0, 6), "happy": slice(6, 12), "neutral": slice(12, 18)}
ACTORS = ["Actor01", "Actor02", "Actor04", "Actor07", "Actor08", "Actor11"]


def corr_distance_rdm(X):
    """Identical to extract_model_rdms.corr_distance_rdm."""
    Xc = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    rdm = 1.0 - (Xc @ Xc.T) / np.maximum(n * n.T, 1e-12)
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def self_test(rng=np.random.default_rng(0)):
    """Slicing the full RDM == recomputing on the subset alone."""
    worst = 0.0
    for _ in range(20):
        X = rng.standard_normal((18, 512)).astype(np.float32)
        full = corr_distance_rdm(X)
        for name, sl in EMOTIONS.items():
            worst = max(worst, np.abs(full[sl, sl] - corr_distance_rdm(X[sl])).max())
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./analysis/model_rdms_canonical")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        worst = self_test()
        print(f"[self-test] max |slice - recompute| = {worst:.3e}")
        if worst > 1e-5:
            sys.exit("ABORT: slicing is not equivalent to recomputation")

    d = Path(args.dir)
    suffixes = tuple(f"_{e}.npy" for e in EMOTIONS)
    srcs = sorted(p for p in d.glob("*.npy") if not p.name.endswith(suffixes))
    print(f"[split] {len(srcs)} source RDMs in {d}")

    for p in srcs:
        r = np.load(p)
        if r.ndim != 4 or r.shape[-2:] != (18, 18):
            print(f"  SKIP {p.name}: shape {r.shape} is not (L,16,18,18)")
            continue
        base = p.with_suffix("")
        for name, sl in EMOTIONS.items():
            block = r[:, :, sl, sl]
            assert block.shape == (r.shape[0], r.shape[1], 6, 6), block.shape
            np.save(f"{base}_{name}.npy", block.astype(np.float32))
        (d / f"{base.name}_byemotion.json").write_text(json.dumps(
            {"source": p.name, "source_shape": list(r.shape),
             "emotion_shape": [r.shape[0], r.shape[1], 6, 6],
             "emotions": {k: [v.start, v.stop] for k, v in EMOTIONS.items()},
             "actor_order": ACTORS}, indent=2))
        print(f"  {p.name} {r.shape} -> 3 x ({r.shape[0]}, {r.shape[1]}, 6, 6)")

    print(f"[ok] wrote {3 * len(srcs)} emotion RDMs")


if __name__ == "__main__":
    main()
