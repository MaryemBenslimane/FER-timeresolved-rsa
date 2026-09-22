"""Standard, model-agnostic activation extraction + prediction + RDM computation.

Works with ANY checkpoint file:
  1. build a known architecture (torchvision by name, or a registered custom builder),
  2. load the checkpoint forgivingly (strip 'module.', strict=False, report missing/extra),
  3. hook every TRAINABLE layer (modules carrying learnable parameters),
  4. run the 18 canonical stimuli x 16 frames, record per-frame class predictions,
  5. compute one correlation-distance RDM per (trainable layer, frame) -> (n_layer,16,18,18).

Stimuli: hdf5_16frames/Slide{1..16}.h5 ['images'] = (18,224,224,3) uint8 RGB, canonical
emotion-major order (fear 0-5, happy 6-11, neutral 12-17; actors 01,02,04,07,08,11).

Examples
  # torchvision resnet50, ImageNet preproc, weights from a checkpoint, auto head size:
  python extract_model_rdms.py --arch resnet50 --checkpoint net_weights/final_resnet50_1000 \
      --preproc imagenet --out-name resnet50_final_16f

  # vgg16_bn with a face checkpoint, VGG-Face caffe preproc:
  python extract_model_rdms.py --arch vgg16_bn --checkpoint net_weights/vgg16_bn_without-meg_pretrained \
      --preproc vggface --out-name vgg16bn_nomeg_pretrained_16f

  # random control (no checkpoint):
  python extract_model_rdms.py --arch vgg16_bn --random --seed 1 --out-name vgg16bn_random_seed1

Outputs (analysis/model_rdms_canonical/):
  <out-name>.npy              (n_trainable_layer, 16, 18, 18) float32 RDMs
  <out-name>.layers.json      layer order + shapes + provenance
  <out-name>.predictions.csv  stim,frame,pred_idx  (per-frame argmax of the model output)
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tvm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.join(_HERE, "image"))   # FaceNet.py lives in models/image/

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data lives directly under scratch (not under a FER/ subdirectory); override
# with FER_ROOT / STIM_DIR, or per-run with --h5-dir / --out-dir.
ROOT = Path(os.environ.get("FER_ROOT", "."))
H5DIR = Path(os.environ.get("STIM_DIR", ROOT / "stim_frames"))
OUT_DIR = ROOT / "analysis/model_rdms_canonical"

# canonical 18 stimuli (emotion-major, numeric actors) -- for prediction labels only
ACTORS = [1, 2, 4, 7, 8, 11]
STIMS = [f"{e}_Actor{a:02d}" for e in ("fear", "happy", "neutral") for a in ACTORS]

# ---------------------------------------------------------------------------
# 1. Architecture registry: name -> builder(num_classes) -> nn.Module
#    Head is replaced to `num_classes` so the checkpoint's task head fits.
# ---------------------------------------------------------------------------
def _resnet(fn, head_dropout=False):
    def build(nc):
        m = fn(weights=None)
        if nc:
            # The net_weights/ checkpoints carry a plain Linear head named
            # `classifier`; wrapping it in Sequential(Dropout, Linear) would
            # rename the keys to fc.1.* and silently fail to load.
            m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, nc)) \
                if head_dropout else nn.Linear(m.fc.in_features, nc)
        return m
    return build

def _vgg(fn):
    def build(nc):
        m = fn(weights=None)
        if nc:
            m.classifier[6] = nn.Linear(m.classifier[6].in_features, nc)
        return m
    return build

def _mobilenet_v2(nc):
    m = tvm.mobilenet_v2(weights=None)
    if nc:
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    return m

def _inception_v3(nc):
    m = tvm.inception_v3(weights=None, aux_logits=True, init_weights=False)
    if nc:
        m.fc = nn.Linear(m.fc.in_features, nc)
    m.aux_logits = False
    return m

def _densenet121(nc):
    m = tvm.densenet121(weights=None)
    if nc:
        m.classifier = nn.Linear(m.classifier.in_features, nc)
    return m

def _facenet(nc):
    # repo FaceNet (models/FaceNet.py): InceptionResnetV1 with last_linear(1792->512)
    # + logits(512->num_classes). n_input_channels=3 for RGB stimuli.
    from FaceNet import InceptionResnetV1 as FaceNetIRV1
    return FaceNetIRV1(num_classes=nc or 2622, n_input_channels=3)

ARCHS = {
    "resnet18": _resnet(tvm.resnet18), "resnet34": _resnet(tvm.resnet34),
    "resnet50": _resnet(tvm.resnet50), "resnet101": _resnet(tvm.resnet101),
    "vgg16": _vgg(tvm.vgg16), "vgg16_bn": _vgg(tvm.vgg16_bn),
    "vgg19": _vgg(tvm.vgg19), "vgg19_bn": _vgg(tvm.vgg19_bn),
    "mobilenet_v2": _mobilenet_v2, "inception_v3": _inception_v3,
    "densenet121": _densenet121, "facenet": _facenet,
}

# ---------------------------------------------------------------------------
# 2. Preprocessing registry: name -> (uint8 NHWC RGB) -> torch NCHW
# ---------------------------------------------------------------------------
def _pp_imagenet(x):
    x = x.astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)

def _pp_vggface(x):                       # caffe-style: BGR, mean-sub, no /255
    x = x.astype(np.float32)[..., ::-1] - np.array([93.5940, 104.7624, 129.1863], np.float32)
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)

def _pp_vggface_rgb(x):
    """RGB mean subtraction used by the AffectNet VGG-Face fine-tuning run."""
    x = x.astype(np.float32) - np.array([129.1863, 104.7624, 93.5940], np.float32)
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)

def _pp_facenet(x):                        # FaceNet fixed standardization: (x-127.5)/128
    x = (x.astype(np.float32) - 127.5) / 128.0
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)

def _pp_none(x):                          # raw 0-1, no normalization
    x = x.astype(np.float32) / 255.0
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)

def _pp_half(x):                          # (x/255 - 0.5)/0.5, as train.py normalises
    x = (x.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)

def _pp_gray_fer(x):
    """Single-channel, matching utils.load_data_FER.Stimuliloader (mean .3612, std .3056).

    That loader relies on the source HDF5 being 2-D so PIL yields mode 'L'.
    stim_frames/ is RGB, so do the ITU-R 601-2 luminance conversion PIL would
    have done. Required by the 1-channel `*_1D_*` checkpoints.
    """
    g = (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114).astype(np.float32) / 255.0
    g = (g - 0.3612) / 0.3056
    return torch.tensor(g[:, None, :, :].copy(), dtype=torch.float32)

PREPROCS = {"imagenet": _pp_imagenet, "vggface": _pp_vggface,
            "vggface_rgb": _pp_vggface_rgb,
            "facenet": _pp_facenet, "none": _pp_none,
            "half": _pp_half, "gray_fer": _pp_gray_fer}


def resize_batch(x, size):
    """Resize a (N,H,W,3) uint8 batch; models trained at 112 should see 112."""
    from PIL import Image
    return np.stack([np.array(Image.fromarray(im).resize((size, size), Image.BILINEAR))
                     for im in x])

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def infer_num_classes(state):
    """Guess head size = out_features of the deepest 2-D Linear/fc weight in the ckpt."""
    best = None
    for k, v in state.items():
        if hasattr(v, "ndim") and v.ndim == 2 and any(t in k.lower() for t in ("fc", "classifier", "logits", "head", "linear")):
            best = v.shape[0]
    # fallback: any 2-D weight
    if best is None:
        for k, v in state.items():
            if hasattr(v, "ndim") and v.ndim == 2:
                best = v.shape[0]
    return best

def load_checkpoint(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj
    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            state = obj[key]; break
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

# ---------------------------------------------------------------------------
# Checkpoint key layout. The net_weights/ checkpoints wrap the torchvision
# backbone in a `model.` attribute, rename ResNet layer{N} -> block{N}, and name
# the head `classifier`. Without remapping, every key is "unexpected", and
# strict=False leaves a randomly initialised network whose RDMs look plausible
# but mean nothing.
# ---------------------------------------------------------------------------
def _repo_to_torchvision(arch, k):
    if arch.startswith("resnet"):
        if k.startswith("classifier."):
            return k.replace("classifier.", "fc.", 1)
        k = re.sub(r"^model\.", "", k)
        return re.sub(r"^block(\d)\.", lambda m: f"layer{m.group(1)}.", k)
    if arch.startswith("vgg"):
        return re.sub(r"^model\.", "features.", k)
    return k                                  # facenet checkpoints are native


def remap_state(arch, state, layout):
    """layout: 'auto' (detect), 'repo' (force remap), 'torchvision' (leave alone)."""
    if layout == "torchvision":
        return state
    if layout == "auto" and not any(k.startswith("model.") for k in state):
        return state
    return {_repo_to_torchvision(arch, k): v for k, v in state.items()}


def randomize(model, seed):
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
            if hasattr(m, "running_mean") and m.running_mean is not None:
                m.running_mean.zero_(); m.running_var.fill_(1.0)
    return model

def trainable_layers(model):
    """Leaf modules that carry learnable parameters (Conv/Linear/Norm/...)."""
    out = []
    for name, mod in model.named_modules():
        if len(list(mod.children())):           # not a leaf
            continue
        if any(p.requires_grad for p in mod.parameters(recurse=False)) or \
           len(list(mod.parameters(recurse=False))) > 0:
            out.append((name, mod))
    return out

def set_in_channels(model, n):
    """Rebuild the first Conv2d to take n input channels.

    The `*_1D_*` checkpoints are trained on grayscale, so their first conv has
    shape (out,1,k,k); loading them into a 3-channel model is a hard size
    mismatch that even strict=False will not tolerate.
    """
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            if mod.in_channels == n:
                return model
            new = nn.Conv2d(n, mod.out_channels, mod.kernel_size, mod.stride,
                            mod.padding, mod.dilation, mod.groups, mod.bias is not None)
            parent, parts = model, name.split(".")
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
            if parts[-1].isdigit():
                parent[int(parts[-1])] = new
            else:
                setattr(parent, parts[-1], new)
            return model
    return model


def weight_layers(model):
    """Conv/Linear leaves only -- the layers an architecture's depth number counts.

    BatchNorm is excluded: it applies a per-channel affine to the output of the
    conv immediately before it, so its RDM duplicates that conv's RDM (measured
    r ~ 0.998 on these stimuli). Hooking both roughly doubles the layer count
    without adding representational information.
    """
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))]


def corr_distance_rdm(X):
    Xc = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    rdm = 1.0 - (Xc @ Xc.T) / np.maximum(n * n.T, 1e-12)
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)

def load_slide(s):
    with h5py.File(H5DIR / f"Slide{s}.h5", "r") as f:
        return np.array(f["images"])            # (18,224,224,3) uint8

# ---------------------------------------------------------------------------
def main():
    global H5DIR, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS), help="architecture to build")
    ap.add_argument("--checkpoint", default=None, help="weights file (state_dict); omit with --random")
    ap.add_argument("--random", action="store_true", help="randomize weights instead of loading a checkpoint")
    ap.add_argument("--seed", type=int, default=42, help="random-init seed (with --random)")
    ap.add_argument("--num-classes", type=int, default=None, help="head size; inferred from checkpoint if omitted")
    ap.add_argument("--preproc", default="imagenet", choices=sorted(PREPROCS), help="input preprocessing")
    ap.add_argument("--n-slides", type=int, default=16, help="number of Slide{i}.h5 frames")
    ap.add_argument("--out-name", required=True, help="output basename under model_rdms_canonical/")
    ap.add_argument("--layers", default="trainable", choices=["trainable", "all", "weights"],
                    help="which leaf modules to hook: 'trainable' = every module with "
                         "parameters (includes BatchNorm), 'weights' = Conv/Linear only, "
                         "'all' = every leaf (default: trainable)")
    ap.add_argument("--ckpt-layout", default="auto", choices=["auto", "repo", "torchvision"],
                    help="key naming of the checkpoint (default: auto-detect)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="proceed even if the checkpoint does not fully match the architecture")
    ap.add_argument("--exclude-head", action="store_true",
                    help="drop the final classification layer from the layer-wise RDMs "
                         "(its low-unit-count RDM is not comparable to the conv layers)")
    ap.add_argument("--head-classes", type=int, default=None,
                    help="build the head with N outputs and randomly re-initialise it, "
                         "loading only the backbone from the checkpoint (e.g. 3 for the "
                         "fear/happy/neutral task). The head tensors are dropped from the "
                         "checkpoint; everything else must still match exactly.")
    ap.add_argument("--in-channels", type=int, default=3,
                    help="input channels; use 1 for the grayscale *_1D_* checkpoints")
    ap.add_argument("--resize", type=int, default=None,
                    help="resize stimuli to NxN before preprocessing (e.g. 112 for models "
                         "trained at 112; default: keep the native 224)")
    ap.add_argument("--h5-dir", default=None, help=f"stimulus directory (default: {H5DIR})")
    ap.add_argument("--manifest", default=None,
                    help="JSON stimulus manifest; defaults to stimulus_manifest.json in --h5-dir")
    ap.add_argument("--out-dir", default=None, help=f"output directory (default: {OUT_DIR})")
    args = ap.parse_args()

    if args.h5_dir:
        H5DIR = Path(args.h5_dir)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    manifest_path = Path(args.manifest) if args.manifest else H5DIR / "stimulus_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        stim_names = [row["stimulus"] for row in manifest]
    else:
        manifest = None
        stim_names = STIMS

    # --- num_classes ---
    nc = args.num_classes
    state = None
    if args.checkpoint and not args.random:
        state = load_checkpoint(args.checkpoint)
        state = remap_state(args.arch, state, args.ckpt_layout)
        if args.head_classes is not None:
            nc = args.head_classes
        elif nc is None:
            nc = infer_num_classes(state)
            print(f"[head] inferred num_classes={nc} from checkpoint")

    # --- build ---
    torch.manual_seed(args.seed)          # reproducible re-initialised head
    model = ARCHS[args.arch](nc)
    if args.in_channels != 3:
        model = set_in_channels(model, args.in_channels)
    model = model.to(DEVICE)
    if args.random:
        randomize(model, args.seed)
        print(f"[{args.arch} RANDOM] Xavier-randomized (seed {args.seed})")
    elif state is not None:
        head_dropped = []
        if args.head_classes is not None:
            # Keep the pretrained backbone, discard the checkpoint's task head:
            # its shape cannot match an N-way head, and load_state_dict raises on
            # a size mismatch even with strict=False.
            ref = model.state_dict()
            head_dropped = [k for k in list(state)
                            if k in ref and tuple(state[k].shape) != tuple(ref[k].shape)]
            for k in head_dropped:
                state.pop(k)
            print(f"[head] re-initialised {args.head_classes}-way head "
                  f"(dropped {len(head_dropped)}: {head_dropped}); backbone from checkpoint")
        miss, unexp = model.load_state_dict(state, strict=False)
        miss = [k for k in miss if k not in head_dropped]   # head is missing by design
        n_ref = len(model.state_dict())
        print(f"[{args.arch}] loaded {args.checkpoint}  "
              f"(matched {n_ref - len(miss)}/{n_ref}, missing {len(miss)}, unexpected {len(unexp)})")
        if len(miss) + len(unexp) > 0:
            print(f"   e.g. missing={list(miss)[:3]}  unexpected={list(unexp)[:3]}")
        # A partial load yields a partly random network, whose RDMs are
        # meaningless but look perfectly well-formed. Refuse by default.
        if (miss or unexp) and not args.allow_partial:
            raise SystemExit(
                f"ABORT: checkpoint did not fully match the {args.arch} architecture "
                f"({len(miss)} missing, {len(unexp)} unexpected). The resulting RDMs "
                f"would be computed from partly random weights. Check --ckpt-layout, "
                f"or pass --allow-partial if this is intentional.")
    else:
        print(f"[{args.arch}] no checkpoint + no --random -> untrained default init")
    model.eval()

    # --- hooks ---
    if args.layers == "trainable":
        layers = trainable_layers(model)
    elif args.layers == "weights":
        layers = weight_layers(model)
    else:
        layers = [(n, m) for n, m in model.named_modules() if not len(list(m.children()))]
    if args.exclude_head and layers:
        # The classification head is the last hooked module. With a small head
        # (e.g. 3 units) its RDM is a low-rank readout whose scale and rank are
        # not comparable to the conv layers, so drop it from the layer-wise set.
        dropped_head_layer = layers[-1][0]
        layers = layers[:-1]
        print(f"[hooks] excluding head layer '{dropped_head_layer}' from the layer-wise RDMs")
    else:
        dropped_head_layer = None
    print(f"[hooks] {len(layers)} {args.layers} layers")
    acts = {}
    handles = []
    for name, mod in layers:
        def mk(k):
            def hook(_m, _i, o):
                o = o[0] if isinstance(o, (tuple, list)) else o
                acts[k] = o.detach().cpu().float().numpy().reshape(o.size(0), -1)
            return hook
        handles.append(mod.register_forward_hook(mk(name)))

    pp = PREPROCS[args.preproc]
    per_slide, order, preds = [], None, []
    with torch.no_grad():
        for s in range(1, args.n_slides + 1):
            arr = load_slide(s)
            if args.resize:
                arr = resize_batch(arr, args.resize)
            x = pp(arr).to(DEVICE)
            acts.clear()
            out = model(x)
            out = out[0] if isinstance(out, (tuple, list)) else out
            if order is None:
                order = list(acts.keys())
            per_slide.append(np.stack([corr_distance_rdm(acts[k]) for k in order], 0))
            pr = out.argmax(1).cpu().numpy() if out.ndim == 2 else np.full(len(x), -1)
            if len(stim_names) != len(pr):
                raise RuntimeError(f"manifest has {len(stim_names)} stimuli but slide has {len(pr)}")
            for i, st in enumerate(stim_names):
                preds.append((st, s, int(pr[i])))
            print(f"  slide {s}: {len(order)} layers", flush=True)
    for h in handles:
        h.remove()

    rdms = np.stack(per_slide, 1)               # (L,16,18,18)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.out_name}.npy"
    np.save(out, rdms)
    (OUT_DIR / f"{args.out_name}.layers.json").write_text(json.dumps(
        {"layer_order": order, "shape": list(rdms.shape), "arch": args.arch,
         "checkpoint": args.checkpoint, "random": args.random, "seed": args.seed,
         "preproc": args.preproc, "num_classes": nc,
         "resize": args.resize, "layers": args.layers,
         "in_channels": args.in_channels, "head_classes": args.head_classes,
         "excluded_head_layer": dropped_head_layer,
         "stimuli": stim_names, "stimulus_manifest": str(manifest_path)}, indent=2))
    with (OUT_DIR / f"{args.out_name}.predictions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["stim", "frame", "pred_idx"]); w.writerows(preds)
    print(f"[ok] {rdms.shape} finite={bool(np.isfinite(rdms).all())} -> {out}")
    print(f"[ok] predictions -> {args.out_name}.predictions.csv")


if __name__ == "__main__":
    main()
