"""Video-based model (Qwen-style VLM) -> canonical per-(layer, frame) RDMs.

Video analog of models/extract_model_rdms.py. Runs each of the 18 stimulus VIDEOS
through a video-capable VLM in native video mode (one forward pass per clip), pools the
visual tokens into 16 temporal groups per layer, then computes one correlation-distance
RDM per (layer, frame) ACROSS the 18 stimuli -> (n_layer, 16, 18, 18), the SAME layout
as the image extractor, so `analysis/paper_stats/timelocked_rsa_general.py` consumes it
unchanged.

Reuses the verified Qwen video-token splitting / temporal grouping from
`vlm_extract_video_mode.py` (QwenVideoExtractor).

Stimulus order = canonical emotion-major (fear 0-5, happy 6-11, neutral 12-17;
actors 01,02,04,07,08,11), matching the EEG conditions.

Examples
  python extract_video_model_rdms.py --model_id Qwen/Qwen2.5-VL-7B-Instruct \
      --out-name qwen25vl7b_video_16f
  python extract_video_model_rdms.py --model_id Qwen/Qwen2.5-VL-3B-Instruct \
      --nframes 16 --frame-list --out-name qwen25vl3b_video_16f

Outputs (analysis/model_rdms_canonical/):
  <out-name>.npy              (n_layer, 16, 18, 18) float32 correlation-distance RDMs
  <out-name>.layers.json      shape + provenance (model_id, nframes, hidden_size)
  <out-name>.predictions.csv  stim, frame, pred_label, prediction_text
"""
import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vlm_extract_video_mode import QwenVideoExtractor, DEFAULT_PROMPT, set_seed, extract_emotion  # noqa: E402

ROOT = Path(os.environ.get("FER_ROOT", "."))
VID_DIR = ROOT / "Stims_Videos/Stims_Videos/Stims_Videos"
OUT_DIR = ROOT / "analysis/model_rdms_canonical"

ACTORS = [1, 2, 4, 7, 8, 11]
EMO_KEYS = [("fear", r"fear"), ("happy", r"hap|joy"), ("neutral", r"neu")]


def canonical_videos():
    """Return the 18 videos in emotion-major / numeric-actor order with stim labels."""
    mp4s = list(VID_DIR.glob("*.mp4"))
    ordered = []
    for emo, pat in EMO_KEYS:
        for a in ACTORS:
            match = [p for p in mp4s if re.search(pat, p.stem, re.I)
                     and re.search(rf"actor\s*0*{a}\b", p.stem, re.I)]
            if not match:
                raise FileNotFoundError(f"no video for {emo} actor{a:02d}")
            ordered.append((f"{emo}_Actor{a:02d}", match[0]))
    return ordered                                         # [(stim, path), ...] length 18


def corr_distance_rdm(X):
    """X: (18, hidden) -> (18,18) correlation distance 1 - Pearson."""
    rdm = 1.0 - np.corrcoef(X)
    return np.nan_to_num(rdm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn_implementation", default=None)
    ap.add_argument("--nframes", type=int, default=32,
                    help="frames fed to the model; temporal pairing halves this (32 -> 16 timepoints). "
                         "With --frame-list, nframes = number of canonical frames (use 16).")
    ap.add_argument("--frame-list", dest="frame_list", action="store_true",
                    help="pass duplicated canonical frames -> exactly --nframes temporal groups")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--manifest", default=None,
                    help="54-stimulus JSON manifest; when omitted use legacy 18 videos")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        videos = [(row["stimulus"], Path(row["video"])) for row in manifest]
    else:
        videos = canonical_videos()
    print(f"[stimuli] {len(videos)} videos (canonical order): {[s for s, _ in videos]}")

    ex = QwenVideoExtractor(args.model_id, args.device, args.dtype,
                            args.attn_implementation, random_weights=args.random)
    tmp = Path(tempfile.mkdtemp(prefix="vidrdm_frames_"))

    acts_all, preds = [], []
    for i, (stim, vpath) in enumerate(videos):
        res = ex.infer_video(str(vpath), args.prompt, args.nframes, args.max_new_tokens,
                             frame_list=args.frame_list, tmpdir=str(tmp))
        acts = res["activations"]                          # (T, n_layers, hidden)
        acts_all.append(acts)
        preds.append((stim, res["pred_label"], res["prediction_text"]))
        print(f"[{i+1}/{len(videos)}] {stim}: acts={acts.shape} pred={res['pred_label']}", flush=True)

    # align frame counts across stimuli (defensive), then stack
    T = min(a.shape[0] for a in acts_all)
    L = min(a.shape[1] for a in acts_all)
    stack = np.stack([a[:T, :L] for a in acts_all], axis=0)
    n_stim = len(videos)
    print(f"[stack] {stack.shape}  ({n_stim} stim, {T} frames, {L} layers)")

    # per (layer, frame) correlation-distance RDM across the 18 stimuli -> (L, T, 18, 18)
    rdms = np.zeros((L, T, n_stim, n_stim), dtype=np.float32)
    for l in range(L):
        for t in range(T):
            rdms[l, t] = corr_distance_rdm(stack[:, t, l, :])

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.out_name}.npy"
    np.save(out, rdms)
    (out_dir / f"{args.out_name}.layers.json").write_text(json.dumps(
        {"shape": list(rdms.shape), "model_id": args.model_id, "n_layers": int(L),
         "n_frames": int(T), "hidden_size": int(stack.shape[-1]), "nframes_fed": args.nframes,
         "frame_list": args.frame_list, "random_weights": args.random,
         "stimuli": [s for s, _ in videos]}, indent=2))
    import csv
    with (out_dir / f"{args.out_name}.predictions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["stim", "pred_label", "prediction_text"]); w.writerows(preds)
    print(f"[ok] {rdms.shape} finite={bool(np.isfinite(rdms).all())} -> {out}")


if __name__ == "__main__":
    main()
