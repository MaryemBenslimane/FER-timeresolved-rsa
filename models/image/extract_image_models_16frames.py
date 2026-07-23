"""Pre-fine-tuning baseline model RDMs for ResNet18-AffectNet and VGGFace-AffectNet.

Builds the SAME architecture as the trained model but with the *pre-fine-tuning*
weights (ImageNet ResNet18 / VGGFace2 InceptionResnetV1) and a randomly initialized
task head, then extracts activations with the identical pipeline as the trained
extraction (same Stimuliloader, same layer_types_to_select, correlation-distance
RDM per layer). Output: (n_layer, 15, 18, 18) matching the trained canonical RDM.

Usage:
  python extract_pretrained_baseline.py --model resnet18   # -> resnet18_affectnet_pretrained.npy
  python extract_pretrained_baseline.py --model vggface    # -> vggface_affectnet_pretrained.npy
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules as mod
from torchvision import models

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.load_data_FER import Stimuliloader
from facenet_pytorch import InceptionResnetV1
from FaceNet import FaceNet, Block35, Mixed_6a, Mixed_7a, Block8, Block17

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HDF5 = "/home/maryem/scratch/FER/hdf5"
OUT_DIR = "/home/maryem/scratch/FER/analysis/model_rdms_canonical"

# Identical to models/extract_model_activations.py
LAYER_TYPES = [mod.activation.PReLU, mod.batchnorm.BatchNorm1d, mod.conv.Conv2d,
               mod.linear.Linear, mod.batchnorm.BatchNorm2d, mod.pooling.AdaptiveAvgPool2d,
               mod.activation.ReLU, mod.pooling.MaxPool2d, mod.activation.ReLU6,
               Block35, Mixed_6a, Mixed_7a, Block8, Block17]


class InceptionResnetV1_Custom(InceptionResnetV1):
    """EXACT architecture of the trained VGGFace model (built via pretrained=None,
    378 hookable layers). Backbone weights are copied in separately."""
    def __init__(self, num_classes=3):
        super().__init__(pretrained=None, classify=True, num_classes=num_classes)
        self.last_linear = nn.Identity()
        self.last_bn = nn.Identity()
        self.logits = nn.Linear(1792, num_classes)


def randomize_all_weights(model, seed=42):
    """Re-initialize weights module-by-module (Xavier-uniform for conv/linear,
    identity for normalization layers) matching the fully-random control. Zeroing
    BatchNorm gamma would collapse activations to constants and yield degenerate
    (NaN) RDMs, so normalization layers are set to the identity transform."""
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
            if hasattr(m, "running_mean") and m.running_mean is not None:
                m.running_mean.zero_(); m.running_var.fill_(1.0)
    return model


def build_model(name, finetuned=None, random=False):
    """finetuned=<ckpt path> -> load the AffectNet-fine-tuned weights (FINE-TUNED);
    random=True -> identical architecture, all weights Xavier-randomized (RANDOM control);
    otherwise -> pre-fine-tuning backbone (ImageNet / VGGFace2) + random head.
    All three use the identical architecture so the RDMs share coordinates."""
    if random:
        torch.manual_seed(42)
        if name == "vggface":
            m = InceptionResnetV1_Custom(num_classes=3)
        elif name == "resnet18":
            m = models.resnet18(weights=None)
            m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, 3))
        else:
            raise ValueError(name)
        randomize_all_weights(m, seed=42)
        print(f"[{name} RANDOM] all weights Xavier-randomized (seed 42)")
        return m.to(DEVICE).eval()
    if name == "vggface":
        m = InceptionResnetV1_Custom(num_classes=3)   # arch identical in both modes
        if finetuned:
            state = torch.load(finetuned, map_location="cpu")
            if any(k.startswith("module.") for k in state):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            miss, unexp = m.load_state_dict(state, strict=False)
            print(f"[vggface FT] loaded {finetuned} (missing {len(miss)}, unexpected {len(unexp)})")
        else:
            vgg2 = InceptionResnetV1(pretrained='vggface2')
            sd, src = m.state_dict(), vgg2.state_dict()
            copied = sum(1 for k in sd if k in src and src[k].shape == sd[k].shape)
            for k in sd:
                if k in src and src[k].shape == sd[k].shape: sd[k] = src[k]
            m.load_state_dict(sd)
            print(f"[vggface PRE] copied {copied}/{len(sd)} backbone tensors from VGGFace2")
    elif name == "resnet18":
        if finetuned:
            m = models.resnet18(weights=None)
            m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, 3))
            state = torch.load(finetuned, map_location="cpu")
            if any(k.startswith("module.") for k in state):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            miss, unexp = m.load_state_dict(state, strict=False)
            print(f"[resnet18 FT] loaded {finetuned} (missing {len(miss)}, unexpected {len(unexp)})")
        else:
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, 3))  # random head
    else:
        raise ValueError(name)
    return m.to(DEVICE).eval()


def corr_distance_rdm(X):
    """X: (18, feat) -> (18,18) correlation distance 1 - Pearson r."""
    Xc = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    denom = np.maximum(n * n.T, 1e-12)
    cor = (Xc @ Xc.T) / denom
    rdm = 1.0 - cor
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def extract(name, finetuned=None, random=False, hdf5_dir=None, n_slides=15):
    hdf5_dir = hdf5_dir or HDF5
    model = build_model(name, finetuned=finetuned, random=random)
    acts = {}
    handles = []
    for lname, layer in model.named_modules():
        if type(layer) in LAYER_TYPES:
            def mk(k):
                def hook(_m, _i, o):
                    acts[k] = o.detach().cpu().numpy().reshape(o.size(0), -1)
                return hook
            handles.append(layer.register_forward_hook(mk(lname)))

    per_slide = []   # each (n_layer, 18, 18)
    layer_order = None
    for s in range(1, n_slides + 1):
        loader = Stimuliloader(18, f"{hdf5_dir}/Slide{s}.h5")
        imgs = next(iter(loader)).to(DEVICE)
        if imgs.shape[1] == 1:            # grayscale -> 3ch for these backbones
            imgs = imgs.repeat(1, 3, 1, 1)
        acts.clear()
        with torch.no_grad():
            model(imgs)
        if layer_order is None:
            layer_order = list(acts.keys())
        rdms = np.stack([corr_distance_rdm(acts[k]) for k in layer_order], axis=0)  # (L,18,18)
        per_slide.append(rdms)
        print(f"  slide {s}: {rdms.shape[0]} layers", flush=True)

    for h in handles:
        h.remove()
    stacked = np.stack(per_slide, axis=1)  # (L, 15, 18, 18)
    return stacked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet18", "vggface"])
    ap.add_argument("--finetuned", default=None, help="path to fine-tuned checkpoint; omit for pretrained baseline")
    ap.add_argument("--random", action="store_true", help="fully Xavier-randomized weights (matched-coord random control)")
    ap.add_argument("--hdf5-dir", default=None, help="stimulus slide directory (default: legacy 15-slide hdf5/)")
    ap.add_argument("--n-slides", type=int, default=15, help="number of Slide{i}.h5 files = model-time points")
    ap.add_argument("--out-name", default=None, help="output basename (without .npy); overrides the default naming")
    args = ap.parse_args()
    tag = {"resnet18": "resnet18_affectnet", "vggface": "vggface_affectnet"}[args.model]
    suffix = "random_matched" if args.random else ("finetuned_matched" if args.finetuned else "pretrained")
    base = args.out_name if args.out_name else f"{tag}_{suffix}"
    out = os.path.join(OUT_DIR, f"{base}.npy")
    rdm = extract(args.model, finetuned=args.finetuned, random=args.random,
                  hdf5_dir=args.hdf5_dir, n_slides=args.n_slides)
    print(f"[{args.model}] {suffix} RDM shape = {rdm.shape}")
    np.save(out, rdm)
    print(f"[ok] saved -> {out}")


if __name__ == "__main__":
    main()
