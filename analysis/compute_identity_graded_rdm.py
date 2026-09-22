#!/usr/bin/env python3
"""Graded actor-identity control RDM.

The binary same/different-actor RDM is CONSTANT within a single emotion (each of
the 6 actors appears exactly once there), so its Spearman is undefined and it can
only be used on the full 18x18. This builds a graded version that is defined in
every condition:

  1. embed all 18 stimuli x 16 frames with VGG-Face (fc7, 4096-d)
  2. average each ACTOR's embedding over its 3 emotions x 16 frames
     -> 6 actor vectors, with emotion averaged out
  3. give every stimulus its actor's vector, then take correlation distance

So distance depends ONLY on actor identity: same-actor pairs are exactly 0
(consistent with the binary RDM), and within one emotion the 6 different actors
have graded, non-constant distances.

Caveat worth stating in any write-up: the identity space comes from VGG-Face,
which is itself one of the compared models, so this control is not independent of
that model family in the way the pixel/motion/flow controls are.

Output: analysis/motion_controls/rdms/identity_graded_rdm.npy
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

ROOT = Path(os.environ.get("FER_ROOT", "."))
sys.path.insert(0, str(ROOT / "models"))
OUT = ROOT / "analysis/motion_controls/rdms/identity_graded_rdm.npy"
CKPT = ROOT / "net_weights/vgg16_vggface"

N_ACTORS = 6
ACTORS = [1, 2, 4, 7, 8, 11]


def pp_vggface(x):
    """caffe-style: BGR, mean-subtract, no /255 -- as extract_model_rdms.py does."""
    x = x.astype(np.float32)[..., ::-1] - np.array([93.5940, 104.7624, 129.1863], np.float32)
    return torch.tensor(np.transpose(x, (0, 3, 1, 2)).copy(), dtype=torch.float32)


def correlation_distance_rdm(X):
    Xc = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    rdm = 1.0 - (Xc @ Xc.T) / np.maximum(n * n.T, 1e-12)
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float32)


def main():
    import h5py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tvm.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 2622)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False),
                          strict=True)
    model.eval().to(device)
    # fc7 = classifier[3]; truncate after its ReLU (classifier[4])
    feat = nn.Sequential(model.features, model.avgpool, nn.Flatten(),
                         *list(model.classifier[:5])).eval().to(device)

    stim_dir = ROOT / "stim_frames"
    slides = sorted(stim_dir.glob("Slide*.h5"))
    embs = []
    with torch.no_grad():
        for p in slides:
            with h5py.File(p, "r") as f:
                img = np.asarray(f["images"])                 # (18,224,224,3)
            e = feat(pp_vggface(img).to(device)).cpu().numpy()  # (18, 4096)
            embs.append(e)
    E = np.stack(embs, 1)                                     # (18, n_frames, 4096)
    print(f"embeddings {E.shape}")

    # average each actor over its 3 emotions and all frames
    actor_of = np.arange(18) % N_ACTORS                       # emotion-major order
    actor_vec = np.stack([E[actor_of == a].mean(axis=(0, 1)) for a in range(N_ACTORS)])
    feats = actor_vec[actor_of]                               # (18, 4096)

    rdm = correlation_distance_rdm(feats)
    iu6 = np.triu_indices(6, 1)
    for name, sl in (("fear", slice(0, 6)), ("happy", slice(6, 12)),
                     ("neutral", slice(12, 18))):
        b = rdm[sl, sl][iu6]
        print(f"  {name:8s} 6x6 off-diag: min={b.min():.4f} max={b.max():.4f} "
              f"std={b.std():.4f}  (binary version was constant)")
    same = rdm[np.triu_indices(18, 1)][
        (actor_of[:, None] == actor_of[None, :])[np.triu_indices(18, 1)]]
    print(f"  same-actor pairs: n={len(same)} max distance={same.max():.2e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, rdm)
    print(f"[ok] saved -> {OUT}")


if __name__ == "__main__":
    main()
