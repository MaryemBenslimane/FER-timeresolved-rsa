#!/usr/bin/env python3
"""M3DFEL (CVPR 2023, Tencent TFace) video RDMs, pretrained and random.

M3DFEL = r3d_18 backbone (Kinetics-400) -> per-instance 512-d -> BiLSTM ->
multi-head self-attention + DMIN -> pwconv -> fc(7). See
M3DFEL/ (github.com/Tencent/TFace/tree/master/attribute/M3DFEL, with our 3-class DFEW changes).

IMPORTANT - what "pretrained" can mean here
    Tencent publishes NO DFEW-trained checkpoint (the repo has zero releases and
    the README only documents training). So --pretrained builds exactly what the
    repo's own code builds: a Kinetics-400 pretrained r3d_18 backbone with a
    randomly initialised MIL head. --random additionally randomises the backbone,
    so the pair isolates the value of the Kinetics-pretrained video features.

Getting 16 time points out of a 4-instance architecture
    The model natively splits num_frames into num_frames/instance_length
    instances, i.e. 16/4 = 4 time points - too coarse to time-lock to the 16 EEG
    windows. Instead we feed 16 OVERLAPPING 4-frame windows (window t covers
    canonical frames t-1..t+2, clipped), so bag_size = 16 while every instance
    still sees genuine 3-D motion. Tensor shapes are exactly those the
    architecture expects.

Stimuli are the SAME canonical 16 frames every other model saw (stim_frames/
Slide{k}.h5), resized to 112 and scaled to [0,1] - the repo's test-time
transform (GroupResize + ToTorchFormatTensor, no mean/std normalisation).

Output: analysis/model_rdms_canonical/<out-name>.npy  (n_layers, 16, 18, 18)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(os.environ.get("FER_ROOT", "."))
# M3DFEL code (bundled in M3DFEL/) and its pretrained/r3d_18-b3b3357e.pth backbone weights.
M3DFEL_DIR = Path(os.environ.get("M3DFEL_DIR", Path(__file__).resolve().parents[1] / "M3DFEL"))
sys.path.insert(0, str(M3DFEL_DIR))

ACTORS = [1, 2, 4, 7, 8, 11]
STIMS = [f"{e}_Actor{a:02d}" for e in ("fear", "happy", "neutral") for a in ACTORS]
INSTANCE_LENGTH = 4


def load_canonical_frames(frames_dir, size=112):
    """(N, 16, 3, size, size) float32 in [0,1] from stimulus-video samples."""
    import h5py
    from PIL import Image
    stim_dir = Path(frames_dir)
    slides = sorted(stim_dir.glob("Slide*.h5"), key=lambda p: int(p.stem.replace("Slide", "")))
    per_frame = []
    for p in slides:
        with h5py.File(p, "r") as f:
            img = np.asarray(f["images"])                      # (18,224,224,3) uint8
        out = np.stack([np.array(Image.fromarray(img[s]).resize((size, size), Image.BILINEAR))
                        for s in range(img.shape[0])])
        per_frame.append(out)
    X = np.stack(per_frame, 1).astype(np.float32) / 255.0       # (18,16,H,W,3)
    return np.transpose(X, (0, 1, 4, 2, 3))                     # (18,16,3,H,W)


def overlapping_windows(x, instance_length=INSTANCE_LENGTH):
    """(16,3,H,W) -> (16*instance_length,3,H,W): window t = frames t-1..t+2, clipped."""
    T = x.shape[0]
    off = np.arange(instance_length) - 1
    idx = np.clip(np.arange(T)[:, None] + off[None, :], 0, T - 1).reshape(-1)
    return x[idx]


def temporal_input(x, mode="original", shuffle_seed=1001, static_frame=0):
    """Transform a canonical clip before making overlapping model instances.

    A common permutation is used across stimuli to preserve matched frame
    sampling. Static clips repeat one source frame across the entire input.
    """
    if mode == "original":
        return x
    if mode == "shuffled":
        return x[np.random.default_rng(shuffle_seed).permutation(len(x))]
    if mode == "static":
        if not 0 <= static_frame < len(x):
            raise ValueError("static frame is outside the canonical clip")
        return np.repeat(x[static_frame:static_frame + 1], len(x), axis=0)
    raise ValueError(f"unknown temporal mode: {mode}")


def randomize(model, seed):
    """Same scheme as extract_model_rdms.py: Xavier conv/linear, identity norms."""
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.GroupNorm, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
            if getattr(m, "running_mean", None) is not None:
                m.running_mean.zero_(); m.running_var.fill_(1.0)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
    return model


def corr_distance_rdm(X):
    Xc = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    rdm = 1.0 - (Xc @ Xc.T) / np.maximum(n * n.T, 1e-12)
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true",
                    help="randomise every weight (default: Kinetics-pretrained backbone)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default=None,
                    help="trained M3DFEL state dict (e.g. selected Fold 3 best_weights.pt)")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "analysis/model_rdms_canonical"))
    ap.add_argument("--frames-dir", default=str(ROOT / "stim_frames"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--temporal-mode", choices=["original", "shuffled", "static"],
                    default="original")
    ap.add_argument("--shuffle-seed", type=int, default=1001)
    ap.add_argument("--static-frame", type=int, default=0,
                    help="zero-based canonical frame repeated for static input")
    args = ap.parse_args()

    from models.M3DFEL import M3DFEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_time = 16
    margs = SimpleNamespace(
        num_frames=n_time * INSTANCE_LENGTH,
        instance_length=INSTANCE_LENGTH,
        num_classes=3 if args.checkpoint else 7,
        gpu_ids=[0] if device.type == "cuda" else [],
        r3d_weights=str(M3DFEL_DIR / "pretrained/r3d_18-b3b3357e.pth"),
    )
    torch.manual_seed(args.seed)
    model = M3DFEL(margs)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        # Training uses 16 frames / 4-frame instances => bag_size=4.  The
        # time-locked extractor uses 16 overlapping instances so upstream
        # representations align one-to-one with the 16 EEG windows.  Only
        # pwconv (the final temporal collapse, which is not retained as an RDM
        # layer) depends on bag_size. Repeat its four learned input-channel
        # kernels to permit the expanded forward pass; all RDM-producing
        # backbone/MIL parameters remain exact checkpoint weights.
        if state["pwconv.weight"].shape != model.pwconv.weight.shape:
            old = state["pwconv.weight"]
            if model.pwconv.weight.shape[1] % old.shape[1] != 0:
                raise RuntimeError("cannot expand trained pwconv to extractor bag size")
            state["pwconv.weight"] = old.repeat_interleave(
                model.pwconv.weight.shape[1] // old.shape[1], dim=1)
            print(f"[time-lock] expanded pwconv {tuple(old.shape)} -> "
                  f"{tuple(state['pwconv.weight'].shape)}; pwconv is excluded from RDMs")
        model.load_state_dict(state, strict=True)
        print(f"[M3DFEL TRAINED] loaded {args.checkpoint}")
    elif args.random:
        model = randomize(model, args.seed)
        print(f"[M3DFEL RANDOM] every weight re-initialised (seed {args.seed})")
    else:
        print("[M3DFEL] Kinetics-400 pretrained r3d_18 backbone + untrained MIL head")
    model.eval().to(device)

    # ---- hooks: every module whose output keeps the instance (time) axis ----
    acts = {}

    def make_hook(name):
        def hook(_m, _i, o):
            o = o[0] if isinstance(o, tuple) else o
            if not torch.is_tensor(o):
                return
            t = o.detach().float()
            if t.dim() == 5:                    # (bag, C, T, H, W) -> (bag, C)
                t = t.mean(dim=(2, 3, 4))
            elif t.dim() == 4:                  # (bag, C, H, W)
                t = t.mean(dim=(2, 3))
            elif t.dim() == 3 and t.shape[0] == 1:   # (1, bag, C)
                t = t[0]
            elif t.dim() == 2:                  # (bag, C)
                pass
            else:
                return
            if t.shape[0] != n_time:
                return
            acts[name] = t.cpu().numpy()
        return hook

    handles, layer_names = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv3d, nn.Conv1d, nn.Linear, nn.LSTM)) or name == "norm":
            handles.append(mod.register_forward_hook(make_hook(name)))
            layer_names.append(name)
    print(f"[hooks] {len(layer_names)} candidate layers")

    X = load_canonical_frames(args.frames_dir)
    manifest_path = Path(args.manifest) if args.manifest else Path(args.frames_dir) / "stimulus_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        stim_names = [row["stimulus"] for row in manifest]
    else:
        stim_names = STIMS
    if len(stim_names) != len(X):
        raise RuntimeError(f"manifest has {len(stim_names)} stimuli but frames contain {len(X)}")
    print(f"[stim] {X.shape}")

    per_stim, order, predictions = [], None, []
    with torch.no_grad():
        for s in range(X.shape[0]):
            transformed = temporal_input(X[s], args.temporal_mode,
                                         args.shuffle_seed, args.static_frame)
            clip = overlapping_windows(transformed)               # (64,3,112,112)
            xb = torch.from_numpy(clip).unsqueeze(0).to(device)    # (1,64,3,112,112)
            acts.clear()
            output = model(xb)
            pred_idx = int(output.argmax(1).item())
            label_order = ["happy", "fear", "neutral"] if args.checkpoint else None
            pred_label = label_order[pred_idx] if label_order and pred_idx < 3 else str(pred_idx)
            predictions.append((stim_names[s], pred_idx, pred_label))
            if order is None:
                order = [n for n in layer_names if n in acts]
                print(f"[hooks] {len(order)} layers kept a 16-point time axis")
            # layers have different channel counts, so keep them separate
            per_stim.append({n: acts[n].copy() for n in order})
    for h in handles:
        h.remove()

    n_stim, L, T = len(per_stim), len(order), n_time
    rdms = np.zeros((L, T, n_stim, n_stim), dtype=np.float32)
    for l, name in enumerate(order):
        A = np.stack([per_stim[s][name] for s in range(n_stim)], 0)   # (18,16,C_l)
        for t in range(T):
            rdms[l, t] = corr_distance_rdm(A[:, t, :].astype(np.float64))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.out_name}.npy", rdms)
    with (out_dir / f"{args.out_name}.predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stim", "pred_idx", "pred_label"])
        w.writerows(predictions)
    (out_dir / f"{args.out_name}.layers.json").write_text(json.dumps({
        "layer_order": order, "shape": list(rdms.shape), "arch": "M3DFEL (r3d_18 + MIL)",
        "random": bool(args.random), "seed": args.seed,
        "checkpoint": args.checkpoint,
        "pretrained_note": ("DFEW 3-class trained checkpoint" if args.checkpoint else
                            ("all weights random" if args.random else
                             "Kinetics-400 r3d_18 backbone; MIL head untrained")),
        "instance_length": INSTANCE_LENGTH, "n_timepoints": T,
        "temporal_mode": args.temporal_mode,
        "shuffle_seed": args.shuffle_seed if args.temporal_mode == "shuffled" else None,
        "frame_order": (np.random.default_rng(args.shuffle_seed).permutation(n_time).tolist()
                        if args.temporal_mode == "shuffled" else
                        [args.static_frame] * n_time if args.temporal_mode == "static" else
                        list(range(n_time))),
        "temporal_scheme": "16 overlapping 4-frame windows (t-1..t+2, clipped)",
        "checkpoint_temporal_adaptation": (
            "trained bag_size-4 pwconv repeated to bag_size 16 only to complete "
            "the expanded forward pass; pwconv/fc are not retained as RDM layers"
            if args.checkpoint else None),
        "stim_order": stim_names, "stimulus_manifest": str(manifest_path)}, indent=2))

    off = ~np.eye(n_stim, dtype=bool)
    v = rdms[:, :, off]
    degen = int((rdms.reshape(L * T, -1).std(1) < 1e-9).sum())
    print(f"[ok] {rdms.shape} finite={bool(np.isfinite(rdms).all())} "
          f"range=[{v.min():.3f},{v.max():.3f}] degenerate={degen}/{L * T}")
    print(f"[ok] -> {out_dir / (args.out_name + '.npy')}")


if __name__ == "__main__":
    main()
