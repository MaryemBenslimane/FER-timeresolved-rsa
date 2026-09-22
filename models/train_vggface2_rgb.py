"""Train VGG16-BN / ResNet-50 / ResNet-18 on an RGB subset of VGGFace2.

Fills the gap in ``net_weights/``: every VGGFace-trained checkpoint there is
1-channel (``*_weights_1D_input_*``, first conv ``[64, 1, 7, 7]``) except
FaceNet.  These runs produce the 3-channel RGB equivalents.

The image subset is built ONCE and cached to a manifest, then reused by every
architecture -- all three models must see identical data or the downstream RSA
comparison between them is confounded.

Checkpoints are saved as plain torchvision ``state_dict``s (no ``model.``
prefix), which is what ``extract_pretrained_baseline.py`` loads into.

Usage::

    python train_vggface2_rgb.py \\
        --data-root /path/to/VGGFace2/train \\
        --arch resnet50 \\
        --num-identities 4605 --max-images 800000 \\
        --epochs 30 --batch-size 128

On "compatible with the existing VGGFace checkpoints": a state_dict stores no
class names, so the exact 4605 identities and their label order used by
``*_0.01LR_32Batch_4605_VGGFaceZ`` are NOT recoverable from ``net_weights/``,
and no code defining that split exists in this tree.  Label order is irrelevant
downstream -- ``extract_pretrained_baseline.py`` discards the identity head and
rebuilds a 3-class one -- so matching the identity COUNT (4605) plus the RGB
input is what actually governs comparability.  We therefore take the first 4605
identities in sorted order.  If the original list turns up, pin it with
``--identity-list`` to reproduce that split exactly.
"""

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMG_EXT = {".jpg", ".jpeg", ".png"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# subset manifest
# --------------------------------------------------------------------------
def build_manifest(data_root: Path, max_images: int, num_identities: int,
                   seed: int, cache: Path, identity_list: Path = None) -> dict:
    """Return ``{'classes': [...], 'train': [[path, label], ...], 'val': [...]}``.

    Cached to ``cache`` so every architecture trains on byte-identical data.
    Scanning VGGFace2 costs minutes on a parallel filesystem; the cache also
    avoids repeating that per job.

    Identity selection and the image budget are independent: pick the identity
    set first (``identity_list``, else the first ``num_identities`` in sorted
    order, else all), then spend ``max_images`` across exactly those
    identities.  Selecting identities by accumulating whole folders until the
    budget ran out would silently yield far fewer than ``num_identities``.

    Classes are sorted, matching ``ImageFolder``'s label convention.  ``seed``
    therefore affects only WHICH images are drawn per identity, never which
    identities are used.
    """
    if cache.exists():
        logger.info("reusing cached manifest %s", cache)
        return json.loads(cache.read_text())

    logger.info("scanning %s (this can take several minutes)", data_root)
    per_id = {}
    for d in sorted(p for p in data_root.iterdir() if p.is_dir()):
        files = sorted(f.name for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        if files:
            per_id[d.name] = files
    logger.info("found %d identities, %d images", len(per_id), sum(map(len, per_id.values())))

    rng = random.Random(seed)

    # ---- 1. identity set -------------------------------------------------
    if identity_list:
        wanted = [l.strip() for l in identity_list.read_text().splitlines() if l.strip()]
        missing = [i for i in wanted if i not in per_id]
        if missing:
            raise SystemExit(
                f"{len(missing)} identities from {identity_list} are absent under "
                f"{data_root} (first few: {missing[:5]}). Refusing to silently train "
                f"on a different identity set.")
        ids = sorted(wanted)
        logger.info("identity set pinned by %s (%d identities)", identity_list, len(ids))
    elif num_identities:
        all_ids = sorted(per_id)
        if num_identities > len(all_ids):
            raise SystemExit(f"asked for {num_identities} identities, only {len(all_ids)} present")
        # first N in sorted order -- deterministic, no seed involved, and the
        # same ordering ImageFolder assigns labels by
        ids = all_ids[:num_identities]
    else:
        ids = sorted(per_id)

    # ---- 2. spend the image budget across exactly those identities -------
    pool = {i: per_id[i][:] for i in ids}
    for i in ids:
        rng.shuffle(pool[i])
    cap = max(1, max_images // len(ids))
    chosen = {i: pool[i][:cap] for i in ids}
    total = sum(len(v) for v in chosen.values())

    # identities poorer than `cap` leave budget unspent -- redistribute it
    # round-robin over identities that still have unused images
    if total < max_images:
        leftovers = {i: pool[i][len(chosen[i]):] for i in ids}
        cursor = 0
        while total < max_images and any(leftovers.values()):
            for i in ids:
                if not leftovers[i]:
                    continue
                chosen[i].append(leftovers[i].pop())
                total += 1
                if total >= max_images:
                    break
            cursor += 1
            if cursor > max_images:      # defensive: never spin forever
                break
    logger.info("selected %d identities, %d images (cap %d/identity)", len(ids), total, cap)

    classes = sorted(chosen)
    cls_idx = {c: i for i, c in enumerate(classes)}

    train, val = [], []
    for c in classes:
        files = chosen[c][:]
        rng.shuffle(files)
        # hold out ~2% per identity (>=1 image) so val covers every class
        n_val = max(1, int(round(0.02 * len(files)))) if len(files) > 1 else 0
        for f in files[:n_val]:
            val.append([f"{c}/{f}", cls_idx[c]])
        for f in files[n_val:]:
            train.append([f"{c}/{f}", cls_idx[c]])

    rng.shuffle(train)
    man = {"classes": classes, "train": train, "val": val,
           "max_images": max_images, "seed": seed,
           "identity_list": str(identity_list) if identity_list else None}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(man))
    logger.info("manifest: %d identities, %d train, %d val -> %s",
                len(classes), len(train), len(val), cache)
    return man


class ListDataset(Dataset):
    """RGB images from an explicit ``(relpath, label)`` list."""

    def __init__(self, root: Path, items, tf):
        self.root, self.items, self.tf = root, items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        # .convert("RGB") is the 3-channel guarantee -- VGGFace2 contains a few
        # genuinely grayscale JPEGs that would otherwise load as 1-channel.
        img = Image.open(self.root / rel).convert("RGB")
        return self.tf(img), label


def inspect_root(data_root: Path, want_identities: int) -> bool:
    """Report whether ``data_root`` has the identity-folder layout we need.

    Written for checking third-party VGGFace2 repackagings (VGGFace2-HQ, the
    112x112 mirrors, Kaggle archives), which often wrap everything in one extra
    top-level directory or ship a flat image dump instead of per-identity dirs.
    Returns True if the layout is usable.
    """
    if not data_root.is_dir():
        logger.error("%s is not a directory", data_root)
        return False

    subdirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    loose = [p for p in data_root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
    logger.info("root %s: %d subdirectories, %d loose image files",
                data_root, len(subdirs), len(loose))

    if not subdirs:
        logger.error("no subdirectories -> not an identity-per-folder layout. "
                     "Point --data-root at the dir whose children are identity folders.")
        return False

    # a single wrapper dir (e.g. archive/VGGface2_HQ/) is the usual repackaging shape
    if len(subdirs) == 1:
        inner = sorted(p for p in subdirs[0].iterdir() if p.is_dir())
        logger.warning("only one subdirectory (%s) containing %d dirs -- you probably "
                       "want --data-root %s", subdirs[0].name, len(inner), subdirs[0])
        return False

    sample = subdirs[:3] + subdirs[len(subdirs) // 2:len(subdirs) // 2 + 1]
    exts, counts = set(), []
    for d in sample:
        files = [f for f in d.iterdir() if f.is_file()]
        counts.append(sum(1 for f in files if f.suffix.lower() in IMG_EXT))
        exts |= {f.suffix.lower() for f in files}
    logger.info("identity folder names: %s ... %s",
                [d.name for d in subdirs[:3]], subdirs[-1].name)
    logger.info("extensions in sampled folders: %s", sorted(exts))
    logger.info("images per sampled folder: %s", counts)

    unusable = exts - IMG_EXT
    if unusable:
        logger.warning("non-image extensions present, will be ignored: %s", sorted(unusable))
    if not any(counts):
        logger.error("sampled identity folders contain no images -- images may be nested "
                     "one level deeper than expected")
        return False

    vgg_style = sum(1 for d in subdirs[:50] if d.name.startswith("n") and d.name[1:].isdigit())
    logger.info("VGGFace2-style 'nNNNNNN' folder names in first 50: %d/50", vgg_style)

    if len(subdirs) < want_identities:
        logger.error("only %d identities present, need %d", len(subdirs), want_identities)
        return False

    try:
        first_img = next(f for d in sample for f in sorted(d.iterdir())
                         if f.suffix.lower() in IMG_EXT)
        with Image.open(first_img) as im:
            logger.info("sample image %s: mode=%s size=%s", first_img.name, im.mode, im.size)
            if im.mode != "RGB":
                logger.warning("sample image is mode=%s; loader converts to RGB anyway", im.mode)
    except StopIteration:
        pass

    logger.info("layout looks usable")
    return True


def make_transforms(size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def build_model(arch: str, num_classes: int) -> nn.Module:
    """Stock torchvision model, RGB (3-channel) input, resized classifier."""
    if arch == "vgg16_bn":
        m = models.vgg16_bn(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """Train (optimizer given) or evaluate; return ``(loss, acc_pct)``."""
    train = optimizer is not None
    model.train(train)
    tot_loss = n_correct = n_total = 0
    t0 = time.time()
    with torch.set_grad_enabled(train):
        for step, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler is not None):
                out = model(x)
                loss = criterion(out, y)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            tot_loss += loss.item() * y.size(0)
            n_correct += (out.argmax(1) == y).sum().item()
            n_total += y.size(0)
            if train and step % 200 == 0:
                logger.info("  step %6d/%d loss %.4f acc %.2f%% (%.1f img/s)",
                            step, len(loader), tot_loss / n_total,
                            100.0 * n_correct / n_total, n_total / (time.time() - t0))
    return tot_loss / n_total, 100.0 * n_correct / n_total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, type=Path, help="VGGFace2 train/ dir")
    p.add_argument("--arch", required=True, choices=["vgg16_bn", "resnet50", "resnet18"])
    p.add_argument("--max-images", type=int, default=800_000)
    p.add_argument("--num-identities", type=int, default=4605,
                   help="first N identities in sorted order; 0 = all. "
                        "4605 matches the existing *_VGGFaceZ checkpoints")
    p.add_argument("--identity-list", type=Path, default=None,
                   help="file with one identity dir name per line; pins the exact "
                        "identity set (use this if the original 4605 list is recovered)")
    p.add_argument("--manifest", type=Path, default=None, help="shared subset cache")
    p.add_argument("--out-dir", type=Path, default=Path("net_weights_rgb"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="inspect the dataset, build the manifest, report whether the "
                        "layout is usable, then exit without training")
    p.add_argument("--resume", type=Path, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ok = inspect_root(args.data_root, args.num_identities)
    if not ok and args.dry_run:
        raise SystemExit("dataset layout unusable as-is (see messages above)")
    if not ok:
        raise SystemExit("refusing to train: dataset layout unusable (run --dry-run for detail)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or args.out_dir / (
        f"subset_{args.max_images}_{args.num_identities}id_seed{args.seed}.json")
    man = build_manifest(args.data_root, args.max_images, args.num_identities,
                         args.seed, manifest, args.identity_list)
    num_classes = len(man["classes"])

    # record the identity set so this run can be reproduced exactly, and so the
    # same identities can be pinned via --identity-list for any later run
    ids_out = manifest.with_suffix(".identities.txt")
    if not ids_out.exists():
        ids_out.write_text("\n".join(man["classes"]) + "\n")
        logger.info("identity set written to %s", ids_out)

    if args.dry_run:
        logger.info("DRY RUN: %d identities, %d train, %d val images -- layout compatible",
                    len(man["classes"]), len(man["train"]), len(man["val"]))
        return

    train_tf, val_tf = make_transforms(args.image_size)
    dl = dict(batch_size=args.batch_size, num_workers=args.workers,
              pin_memory=True, persistent_workers=args.workers > 0)
    train_ld = DataLoader(ListDataset(args.data_root, man["train"], train_tf),
                          shuffle=True, drop_last=True, **dl)
    val_ld = DataLoader(ListDataset(args.data_root, man["val"], val_tf),
                        shuffle=False, **dl)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, num_classes).to(device)
    logger.info("%s | %d identities | %d train / %d val | device %s",
                args.arch, num_classes, len(man["train"]), len(man["val"]), device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.weight_decay,
                                nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = None if args.no_amp or device.type != "cuda" else torch.amp.GradScaler()

    start_epoch, best = 0, 0.0
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best_acc", 0.0)
        logger.info("resumed from %s at epoch %d", args.resume, start_epoch)

    tag = f"{args.arch}_VGGFace2_RGB_{num_classes}id"

    for epoch in range(start_epoch, args.epochs):
        logger.info("epoch %d/%d (lr %.5f)", epoch + 1, args.epochs, sched.get_last_lr()[0])
        tr_loss, tr_acc = run_epoch(model, train_ld, criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_ld, criterion, device)
        sched.step()
        logger.info("epoch %d: train %.4f/%.2f%% | val %.4f/%.2f%%",
                    epoch + 1, tr_loss, tr_acc, va_loss, va_acc)

        # plain torchvision state_dict -- drops into extract_pretrained_baseline.py
        torch.save(model.state_dict(), args.out_dir / tag)
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "best_acc": max(best, va_acc)}, args.out_dir / f"{tag}_resume.pt")
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), args.out_dir / f"{tag}_best")
        (args.out_dir / f"{tag}.meta.json").write_text(json.dumps({
            "arch": args.arch, "in_channels": 3, "num_classes": num_classes,
            "dataset": "VGGFace2", "manifest": str(manifest),
            "identity_list": str(ids_out),
            "n_train": len(man["train"]), "n_val": len(man["val"]),
            "epoch": epoch + 1, "val_acc": va_acc, "best_val_acc": best,
            "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed}, indent=2))

    logger.info("done. best val acc %.2f%% -> %s", best, args.out_dir / f"{tag}_best")


if __name__ == "__main__":
    main()
