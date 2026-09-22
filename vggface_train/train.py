#!/usr/bin/env python3
"""Train one backbone (resnet50 / resnet18 / facenet) on VGGFace2 identity classification.

One process owns one GPU; three of these run concurrently on a node, one per GPU.
Reads the prebuilt index (see build_index.py) so it never walks the dataset tree.
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# --------------------------------------------------------------------------- data

class ListDataset(Dataset):
    """Images listed in a '<relpath>\\t<label>' index file, rooted at `root`."""

    def __init__(self, root, index_file, transform, decode_size):
        self.root = root
        self.transform = transform
        # draft() decodes JPEG straight to the nearest 1/2^n scale, so a 512px
        # source becomes 128px without ever materialising the full-size bitmap.
        # This is the single biggest CPU win in the pipeline.
        self.decode_size = (decode_size, decode_size)
        with open(index_file) as f:
            rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
        self.samples = [(p, int(l)) for p, l in rows]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(os.path.join(self.root, path)) as img:
            img.draft("RGB", self.decode_size)
            img = img.convert("RGB")
            return self.transform(img), label


def build_transforms(size, decode_size):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.111),
                                     interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val = transforms.Compose([
        transforms.Resize(decode_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train, val


# -------------------------------------------------------------------------- model

def build_model(arch, num_classes):
    if arch == "resnet50":
        return torchvision.models.resnet50(weights=None, num_classes=num_classes)
    if arch == "resnet18":
        return torchvision.models.resnet18(weights=None, num_classes=num_classes)
    if arch == "facenet":
        from facenet_pytorch import InceptionResnetV1
        return InceptionResnetV1(pretrained=None, classify=True, num_classes=num_classes)
    raise ValueError(f"unknown arch: {arch}")


def param_groups(model, weight_decay):
    """Keep weight decay off biases and norm parameters."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


# ---------------------------------------------------------------------- train/eval

def accuracy(logits, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(target.view(-1, 1))
    return [correct[:, :k].any(dim=1).float().sum().item() for k in topk]


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, amp):
    model.eval()
    top1 = top5 = n = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            logits = model(images)
        c1, c5 = accuracy(logits.float(), target)
        top1 += c1
        top5 += c5
        n += target.size(0)
    return 100 * top1 / max(n, 1), 100 * top5 / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["resnet50", "resnet18", "facenet"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=None, help="default: 0.1 * batch_size/256")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--size", type=int, default=112)
    p.add_argument("--decode-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--max-hours", type=float, default=None,
                   help="checkpoint and stop cleanly before the SLURM walltime")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", default="auto")
    args = p.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda \
        else torch.float32
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.index_dir, "classes.txt")) as f:
        num_classes = sum(1 for line in f if line.strip())

    train_tf, val_tf = build_transforms(args.size, args.decode_size)
    train_set = ListDataset(args.data_root, os.path.join(args.index_dir, "train.txt"),
                            train_tf, args.decode_size)
    val_set = ListDataset(args.data_root, os.path.join(args.index_dir, "val.txt"),
                          val_tf, args.decode_size)

    def loader(ds, shuffle, bs):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=args.workers,
                          pin_memory=True, drop_last=shuffle,
                          persistent_workers=args.workers > 0,
                          prefetch_factor=4 if args.workers > 0 else None)

    train_loader = loader(train_set, True, args.batch_size)
    val_loader = loader(val_set, False, args.batch_size)

    model = build_model(args.arch, num_classes).to(device, memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)

    lr = args.lr if args.lr is not None else 0.1 * args.batch_size / 256
    optimizer = torch.optim.SGD(param_groups(model, args.weight_decay), lr=lr,
                                momentum=args.momentum, nesterov=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return 0.5 * lr * (1 + math.cos(math.pi * min(progress, 1.0)))

    print(f"[{args.arch}] classes={num_classes} train={len(train_set)} val={len(val_set)} "
          f"steps/epoch={steps_per_epoch} bs={args.batch_size} lr={lr:.4f} amp={amp_dtype}",
          flush=True)

    ckpt_path = os.path.join(args.out_dir, "last.pt")
    best_path = os.path.join(args.out_dir, "best.pt")
    weights_path = os.path.join(args.out_dir, "last_weights.pt")
    best_weights_path = os.path.join(args.out_dir, "best_weights.pt")
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    start_epoch, global_step, best_top1 = 0, 0, 0.0

    if args.resume == "auto" and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_top1 = state["best_top1"]
        print(f"[{args.arch}] resumed from epoch {start_epoch}", flush=True)

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_top1", "val_top1", "val_top5",
                 "lr", "img_per_s", "epoch_s", "elapsed_s"])

    stop_reason = "completed"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        e0 = time.time()
        run_loss = run_top1 = seen = 0

        for i, (images, target) in enumerate(train_loader):
            for g in optimizer.param_groups:
                g["lr"] = lr_at(global_step)

            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=cuda):
                logits = model(images)
                loss = criterion(logits, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = target.size(0)
            run_loss += loss.item() * bs
            run_top1 += accuracy(logits.detach().float(), target, topk=(1,))[0]
            seen += bs
            global_step += 1

            if (i + 1) % args.log_every == 0:
                rate = seen / (time.time() - e0)
                print(f"[{args.arch}] e{epoch} {i + 1}/{steps_per_epoch} "
                      f"loss={run_loss / seen:.3f} top1={100 * run_top1 / seen:.2f}% "
                      f"lr={optimizer.param_groups[0]['lr']:.4f} {rate:.0f} img/s", flush=True)

        epoch_s = time.time() - e0
        val_top1, val_top5 = evaluate(model, val_loader, device, amp_dtype, cuda)
        elapsed = time.time() - t0
        print(f"[{args.arch}] epoch {epoch} done in {epoch_s:.0f}s  "
              f"train_top1={100 * run_top1 / seen:.2f}%  "
              f"val_top1={val_top1:.2f}%  val_top5={val_top5:.2f}%", flush=True)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{run_loss / seen:.4f}", f"{100 * run_top1 / seen:.4f}",
                                    f"{val_top1:.4f}", f"{val_top5:.4f}",
                                    f"{optimizer.param_groups[0]['lr']:.6f}",
                                    f"{seen / epoch_s:.1f}", f"{epoch_s:.1f}", f"{elapsed:.1f}"])

        state = {"arch": args.arch, "epoch": epoch, "global_step": global_step,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "best_top1": max(best_top1, val_top1), "num_classes": num_classes,
                 "args": vars(args)}
        torch.save(state, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        torch.save(state["model"], weights_path + ".tmp")
        os.replace(weights_path + ".tmp", weights_path)
        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(state, best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
            torch.save(state["model"], best_weights_path + ".tmp")
            os.replace(best_weights_path + ".tmp", best_weights_path)

        # Stop on our own terms rather than being killed mid-epoch by SLURM.
        if args.max_hours is not None:
            spent = (time.time() - t0) / 3600
            if spent + (epoch_s + 60) / 3600 > args.max_hours:
                stop_reason = f"time budget ({spent:.2f}h of {args.max_hours}h, next epoch would overrun)"
                print(f"[{args.arch}] stopping: {stop_reason}", flush=True)
                break

    print(f"[{args.arch}] FINISHED ({stop_reason}) best_val_top1={best_top1:.2f}% "
          f"total={(time.time() - t0) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
