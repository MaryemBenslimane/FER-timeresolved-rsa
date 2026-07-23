#!/usr/bin/env python3
"""Qwen2.5-VL **native video mode** activation extraction (Option A).

Contrast with `vlm_extract_activations.py`, which sends each frame as an
INDEPENDENT still image (16 separate forward passes, no motion information).
Here the whole clip goes through in ONE forward pass as a video, so Qwen's 3D
patch embedding and time-aware mRoPE actually see the dynamics.

Recovering a model-time axis from a single pass
-----------------------------------------------
Qwen2.5-VL fuses frames in **temporal pairs** at the 3D patch embedding
(``temporal_patch_size = 2``), then spatially merges patches by ``merge_size``.
The processor reports ``video_grid_thw = (T, H, W)`` in patches, and the visual
tokens are laid out **time-major**:

    [ group 0 tokens ][ group 1 tokens ] ... [ group T-1 tokens ]
      each of size  (H // merge) * (W // merge)

So we locate the video tokens in ``input_ids``, split them into ``T`` contiguous
chunks, and mean-pool each layer's hidden states within each chunk. That yields one
representation per temporal group -- a genuine time axis, with full temporal
attention already applied.

Because ``nframes`` frames -> ``nframes / 2`` temporal groups, pass ``--nframes 32``
to obtain 16 timepoints (matching the frame-mode grid of the other models).

Output (identical layout to the frame-mode extractor, so
`compute_rdms_from_vlm_activations.py` consumes it unchanged):
    activations/{condition_id}.npy   -- (n_timepoints, n_layers, hidden_size)
    frame_predictions.csv            -- one row per timepoint
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import transformers
from transformers import AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # video mode genuinely requires this helper
    raise SystemExit("qwen_vl_utils is required for video mode: pip install qwen-vl-utils") from exc


EMOTION_LABELS = ["neutral", "happy", "fearful"]
DEFAULT_PROMPT = (
    "What is the facial emotion of the person in this video? "
    f"Choose exactly one from: {', '.join(EMOTION_LABELS)}. "
    "Reply with only the label, nothing else."
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_emotion(text: str) -> str:
    t = (text or "").strip().lower()
    for lab in EMOTION_LABELS:
        if lab in t:
            return lab
    if "fear" in t:
        return "fearful"
    if "happ" in t or "joy" in t:
        return "happy"
    return "unknown"


def load_model(model_id: str, device: str, dtype: str, attn_implementation: str | None):
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "[FATAL] --device cuda requested but torch.cuda.is_available() is False. "
            "The allocated GPU failed to initialise; refusing to fall back to CPU "
            "(a 7B video forward pass on CPU takes hours). Resubmit on another node."
        )
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}.get(dtype, torch.bfloat16)
    kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    errors = []
    for cls_name in ("Qwen2_5_VLForConditionalGeneration",
                     "Qwen3VLForConditionalGeneration",
                     "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            model.eval()
            return model, cls_name
        except Exception as exc:  # try the next class
            errors.append(f"{cls_name}: {exc}")
    raise RuntimeError("Could not load a video-capable Qwen class.\n" + "\n".join(errors))


class QwenVideoExtractor:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16",
                 attn_implementation: str | None = None):
        self.model, self.model_class = load_model(model_id, device, dtype, attn_implementation)
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"[model] {model_id} loaded as {self.model_class}", flush=True)

    # -- locating the visual tokens -------------------------------------------------
    def _video_token_id(self) -> int:
        tid = getattr(self.model.config, "video_token_id", None)
        if tid is not None:
            return int(tid)
        for tok in ("<|video_pad|>", "<|vision_pad|>"):
            cand = self.processor.tokenizer.convert_tokens_to_ids(tok)
            if cand is not None and cand >= 0:
                return int(cand)
        raise RuntimeError("Could not determine the video token id.")

    def _merge_size(self) -> int:
        ip = getattr(self.processor, "image_processor", None)
        return int(getattr(ip, "merge_size", 2) or 2)

    @staticmethod
    def canonical_frame_paths(video_path: str, n_frames: int, tmpdir: str) -> List[str]:
        """Sample n_frames with the SAME rule as every other model
        (CAP_PROP_FRAME_COUNT + linspace + seek), write them to disk, and return each
        path TWICE. Qwen fuses frames in temporal pairs, so 2*n_frames images collapse
        to exactly n_frames temporal groups, one per canonical frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = (np.arange(total) if total <= n_frames
               else np.linspace(0, total - 1, n_frames, dtype=int))
        paths: List[str] = []
        stem = Path(video_path).stem
        for k, i in enumerate(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Could not read frame {int(i)} from {video_path}")
            p = os.path.join(tmpdir, f"{stem}_f{k:03d}.jpg")
            cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.extend([p, p])          # duplicate -> one temporal group per frame
        cap.release()
        return paths

    @torch.inference_mode()
    def infer_video(self, video_path: str, prompt: str, nframes: int,
                    max_new_tokens: int, frame_list: bool = False,
                    tmpdir: str | None = None) -> Dict[str, Any]:
        if nframes % 2 != 0 and not frame_list:
            raise ValueError("nframes must be even (Qwen fuses frames in temporal pairs).")

        if frame_list:
            # nframes here = number of CANONICAL frames == number of temporal groups
            vid = self.canonical_frame_paths(video_path, nframes, tmpdir or ".")
        else:
            vid = video_path

        messages = [{
            "role": "user",
            "content": ([{"type": "video", "video": vid}] if frame_list
                        else [{"type": "video", "video": vid, "nframes": nframes}]) + [
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        if not video_inputs:
            raise RuntimeError(f"process_vision_info returned no video for {video_path}")

        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt")
        dev = next(self.model.parameters()).device
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model returned no hidden_states.")

        # --- split the visual tokens into temporal groups ---
        ids = inputs["input_ids"][0]
        vid_positions = (ids == self._video_token_id()).nonzero(as_tuple=True)[0]
        if vid_positions.numel() == 0:
            raise RuntimeError("No video tokens found in input_ids.")

        grid = inputs["video_grid_thw"][0].tolist()          # [T, H, W] in patches
        n_groups = int(grid[0])
        merge = self._merge_size()
        per_group = int(grid[1] * grid[2] // (merge * merge))
        if per_group <= 0 or n_groups * per_group != int(vid_positions.numel()):
            # defensive fallback: split the token run evenly
            per_group = int(vid_positions.numel()) // max(n_groups, 1)
            print(f"  [warn] grid {grid} vs {int(vid_positions.numel())} tokens; "
                  f"falling back to {n_groups}x{per_group}", flush=True)
        usable = n_groups * per_group

        per_layer: List[np.ndarray] = []
        for layer_hidden in hidden_states:                    # (1, seq, hidden)
            vis = layer_hidden[0][vid_positions][:usable]     # (usable, hidden)
            vis = vis.reshape(n_groups, per_group, -1).mean(dim=1)   # (T, hidden)
            vis = F.normalize(vis.float(), dim=-1)            # same normalisation as frame mode
            per_layer.append(vis.cpu().numpy())
        acts = np.stack(per_layer, axis=1)                    # (T, n_layers, hidden)

        gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_only = gen_ids[:, inputs["input_ids"].shape[1]:]
        pred_text = self.processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()

        return {"activations": acts, "n_groups": n_groups, "per_group": per_group,
                "grid_thw": grid, "prediction_text": pred_text,
                "pred_label": extract_emotion(pred_text)}


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen2.5-VL native VIDEO-mode activation extraction.")
    p.add_argument("--metadata_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn_implementation", default=None)
    p.add_argument("--nframes", type=int, default=32,
                   help="Frames fed to the model; temporal pairing halves this "
                        "(32 -> 16 timepoints, matching the frame-mode grid).")
    p.add_argument("--frame-list", dest="frame_list", action="store_true",
                   help="Pass duplicated canonical frames as the video (gives exactly --nframes temporal groups).")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir); (out_dir / "activations").mkdir(parents=True, exist_ok=True)
    act_dir = out_dir / "activations"

    df = pd.read_csv(args.metadata_csv)
    missing = {"video_path", "stimulus_id"} - set(df.columns)
    if missing:
        raise ValueError(f"metadata_csv missing columns: {missing}")
    if args.max_samples > 0:
        df = df.iloc[: args.max_samples].copy()

    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix='qwen_vid_frames_'))
    ex = QwenVideoExtractor(args.model_id, args.device, args.dtype, args.attn_implementation)

    rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        video_path = str(row["video_path"])
        stimulus_id = str(row["stimulus_id"])
        label = row["label"] if "label" in row else None
        cond = f"{label}_{stimulus_id}" if label is not None else stimulus_id
        if not os.path.exists(video_path):
            print(f"[WARN] missing video, skipping: {video_path}"); continue

        res = ex.infer_video(video_path, args.prompt, args.nframes, args.max_new_tokens,
                             frame_list=args.frame_list, tmpdir=str(tmp))
        acts = res["activations"]
        np.save(act_dir / f"{cond}.npy", acts)
        print(f"[{i+1}/{len(df)}] {cond}: acts={acts.shape} grid={res['grid_thw']} "
              f"({res['n_groups']} timepoints x {res['per_group']} tok/group) "
              f"pred={res['pred_label']}", flush=True)

        for t in range(acts.shape[0]):
            rows.append({"condition_id": cond, "stimulus_id": stimulus_id,
                         "video_path": video_path, "frame_index": t,
                         "prediction_text": res["prediction_text"],
                         "pred_label": res["pred_label"],
                         "n_layers": int(acts.shape[1]), "hidden_size": int(acts.shape[2]),
                         **({"label": label} if label is not None else {})})

    if not rows:
        raise RuntimeError("No videos processed - check metadata paths.")
    pd.DataFrame(rows).to_csv(out_dir / "frame_predictions.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(
        {"mode": "video", "model_id": args.model_id, "nframes": args.nframes,
         "n_conditions": len(set(r["condition_id"] for r in rows))}, indent=2))
    print(f"[ok] wrote {act_dir} and frame_predictions.csv")


if __name__ == "__main__":
    main()
