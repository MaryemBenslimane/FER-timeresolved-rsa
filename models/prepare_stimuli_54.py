#!/usr/bin/env python3
"""Build the canonical 54-stimulus/16-frame set from the occlusion videos."""
import csv
import json
from pathlib import Path

import av
import h5py
import numpy as np
from PIL import Image

ROOT = Path(".")
VIDEOS = ROOT / "Stims_Videos/Stims_Videos/Stims_Videos"
OUT = ROOT / "stim_frames_54"
ACTORS = (1, 2, 4, 7, 8, 11)
EMOTIONS = ("fear", "happy", "neutral")
CONDITIONS = ("original", "glasses", "masked")


def video_path(emotion, condition, actor):
    prefix = {"fear": "Fear", "neutral": "Neutral", "happy": "Hapiness"}[emotion]
    if emotion == "happy" and condition == "original" and actor == 11:
        name = "Happiness_Actor11.mp4"  # lone correctly-spelled source filename
    else:
        middle = {"original": "", "glasses": "_Glasses", "masked": "_Masked"}[condition]
        name = f"{prefix}{middle}_Actor{actor:02d}.mp4"
    return VIDEOS / name


def sample_16(path):
    with av.open(str(path)) as container:
        frames = [
            np.asarray(frame.to_image().convert("RGB").resize(
                (224, 224), Image.Resampling.LANCZOS))
            for frame in container.decode(video=0)
        ]
    if not frames:
        raise RuntimeError(f"no video frames decoded: {path}")
    indices = np.rint(np.linspace(0, len(frames) - 1, 16)).astype(int)
    return np.stack([frames[i] for i in indices]), len(frames), indices


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, clips = [], []
    index = 0
    for emotion in EMOTIONS:
        for condition in CONDITIONS:
            for actor in ACTORS:
                path = video_path(emotion, condition, actor)
                if not path.exists():
                    raise FileNotFoundError(path)
                sampled, total, frame_indices = sample_16(path)
                clips.append(sampled)
                manifest.append({
                    "index": index, "emotion": emotion, "condition": condition,
                    "actor": f"{actor:02d}",
                    "stimulus": f"{emotion}_{condition}_Actor{actor:02d}",
                    "video": str(path), "decoded_frames": total,
                    "sampled_frame_indices": frame_indices.tolist(),
                })
                index += 1
    clips = np.stack(clips)  # 54,16,H,W,3
    for frame in range(16):
        with h5py.File(OUT / f"Slide{frame + 1}.h5", "w") as f:
            f.create_dataset("images", data=clips[:, frame], compression="gzip",
                             compression_opts=1)
    (OUT / "stimulus_manifest.json").write_text(json.dumps(manifest, indent=2))
    with (OUT / "stimulus_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "emotion", "condition", "actor", "stimulus", "video",
            "decoded_frames", "sampled_frame_indices"])
        writer.writeheader()
        for row in manifest:
            writer.writerow({**row, "sampled_frame_indices":
                             " ".join(map(str, row["sampled_frame_indices"]))})
    np.save(OUT / "all_video_frames_16.npy", clips)
    print(f"[ok] clips={clips.shape} -> {OUT}")
    print("[order] fear[0:18], happy[18:36], neutral[36:54]; "
          "within each: original, glasses, masked")


if __name__ == "__main__":
    main()
