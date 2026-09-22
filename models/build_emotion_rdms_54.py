#!/usr/bin/env python3
"""Validate 54-stimulus model RDMs and write three 18x18 emotion slices."""
import argparse
import json
from pathlib import Path

import numpy as np

SLICES = {"fear": (0, 18), "happy": (18, 36), "neutral": (36, 54)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    root = Path(args.dir)
    paths = ([root / f"{tag}.npy" for tag in args.models] if args.models else
             sorted(root.glob("*.npy")))
    for path in paths:
        if any(path.stem.endswith("_" + emotion) for emotion in SLICES):
            continue
        rdm = np.load(path)
        if rdm.ndim != 4 or rdm.shape[1:] != (16, 54, 54):
            if args.models:
                raise RuntimeError(f"{path.name}: expected (L,16,54,54), got {rdm.shape}")
            continue
        if not np.isfinite(rdm).all() or not np.allclose(rdm, rdm.swapaxes(-1, -2), atol=1e-5):
            raise RuntimeError(f"{path.name}: non-finite or asymmetric RDM")
        outputs = {}
        for emotion, (start, stop) in SLICES.items():
            out = rdm[:, :, start:stop, start:stop]
            np.save(root / f"{path.stem}_{emotion}.npy", out)
            outputs[emotion] = {"range": [start, stop], "shape": list(out.shape)}
        (root / f"{path.stem}_byemotion.json").write_text(json.dumps({
            "source": path.name,
            "global_order": ["fear", "happy", "neutral"],
            "within_emotion_order": ["original", "glasses", "masked"],
            "outputs": outputs,
        }, indent=2))
        print(f"[ok] {path.name} {rdm.shape} -> 3 x (L,16,18,18)")


if __name__ == "__main__":
    main()
