#!/usr/bin/env python3
"""Build 16-frame stimulus HDF5 "slides" straight from the source videos.

Replaces the legacy 15 `Slide{i}.h5` files (built from pre-extracted PNGs whose
directory contains case-duplicated and mis-spelled variants) with a clean 16-frame
set sampled the SAME way as the DINOv2 / VLM pipelines:

    idx = np.linspace(0, total_frames - 1, 16).astype(int)

Canonical stimulus order = the videos in Stims_Videos, sorted by EMOTION then ACTOR:

    [ 0- 5] fear      Actor 01,02,04,07,08,11
    [ 6-11] happiness Actor 01,02,04,07,08,11
    [12-17] neutral   Actor 01,02,04,07,08,11

Emotion is normalised so that both "Hapiness_*" and "Happiness_*" map to happiness
(the source files are inconsistently spelled), and actors sort numerically so
Happiness_Actor07 lands in actor order rather than after Actor11.

Writes, into --out-dir:
    Slide{1..16}.h5      dataset "images" = (18, size, size, 3) uint8
    stimulus_order.json  the explicit 18-stimulus order + per-video frame indices
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import h5py
import numpy as np

EMOTION_RANK = {"fear": 0, "happy": 1, "neutral": 2}


def normalize_emotion(stem: str) -> str:
    s = stem.lower()
    if s.startswith("fear"):
        return "fear"
    if s.startswith("hap"):          # covers both "Hapiness" and "Happiness"
        return "happy"
    if s.startswith("neutral"):
        return "neutral"
    raise ValueError(f"Unrecognised emotion in filename: {stem!r}")


def actor_number(stem: str) -> int:
    m = re.search(r"actor\s*0*(\d+)", stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"No actor number in filename: {stem!r}")
    return int(m.group(1))


def canonical_videos(video_dir: Path) -> list[Path]:
    vids = sorted(video_dir.glob("*.mp4"))
    if len(vids) != 18:
        raise ValueError(f"Expected 18 videos in {video_dir}, found {len(vids)}")
    vids.sort(key=lambda p: (EMOTION_RANK[normalize_emotion(p.stem)], actor_number(p.stem)))
    return vids


def sample_frames(path: Path, n_frames: int, size: int):
    """Sample frames EXACTLY as the VLM / DINOv2 pipelines do, so the model-time
    axes are identical across models:

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # container metadata
        indices      = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)                   # seek, don't stream

    Using len(decoded_frames) instead of the metadata count would shift the indices
    off the other models' grid.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {path}")

    if total <= n_frames:
        idx = np.arange(total, dtype=int)
    else:
        idx = np.linspace(0, total - 1, n_frames, dtype=int)

    out = np.zeros((len(idx), size, size, 3), dtype=np.uint8)
    for k, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read frame {int(i)} from {path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out[k] = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    cap.release()
    if len(idx) != n_frames:
        raise ValueError(f"{path.name}: got {len(idx)} frames, expected {n_frames}")
    return out, idx.tolist(), total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path,
                    default=Path("./Stims_Videos/Stims_Videos"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("./hdf5_16frames"))
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vids = canonical_videos(args.video_dir)

    print("Canonical stimulus order (emotion -> actor):")
    order = []
    for i, v in enumerate(vids):
        emo, act = normalize_emotion(v.stem), actor_number(v.stem)
        print(f"  [{i:2d}] {v.name:<28s} emotion={emo:<8s} actor={act}")
        order.append({"index": i, "file": v.name, "emotion": emo, "actor": act})

    # (18, n_frames, size, size, 3)
    per_stim = np.zeros((len(vids), args.n_frames, args.size, args.size, 3), dtype=np.uint8)
    meta = []
    for i, v in enumerate(vids):
        arr, idx, total = sample_frames(v, args.n_frames, args.size)
        per_stim[i] = arr
        meta.append({"index": i, "file": v.name, "total_frames": total, "sampled_indices": idx})
        print(f"  {v.name}: {total} native frames -> sampled {idx}", flush=True)

    for t in range(args.n_frames):
        out_path = args.out_dir / f"Slide{t+1}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("images", data=per_stim[:, t])     # (18, size, size, 3)
        print(f"[ok] {out_path.name}  images={per_stim[:, t].shape}")

    (args.out_dir / "stimulus_order.json").write_text(json.dumps(
        {"order": order,
         "emotion_indices": {e: [o["index"] for o in order if o["emotion"] == e]
                             for e in ("fear", "happy", "neutral")},
         "frames": meta,
         "n_frames": args.n_frames, "size": args.size}, indent=2))
    print(f"[ok] wrote stimulus_order.json")
    print("emotion indices:",
          {e: [o['index'] for o in order if o['emotion'] == e] for e in ('fear', 'happy', 'neutral')})


if __name__ == "__main__":
    main()
