#!/usr/bin/env python3
"""Fine-tune the face-pretrained image models on 3-class AffectNet (fear, happy, neutral).

    facenet        InceptionResnetV1, our VGGFace2 pretraining
    resnet18/50    torchvision ResNets, our VGGFace2 pretraining
    vgg16_vggface  VGG-Face (Parkhi et al., 2015; no batch norm)
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


CLASSES = ["fear", "happy", "neutral"]


class VGGFace(nn.Module):
    """VGG-Face network matching the public vgg_face_dag.pth state dict."""
    def __init__(self, num_classes=2622):
        super().__init__()
        layers = []
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
               512, 512, 512, "M", 512, 512, 512, "M"]
        channels = 3
        for value in cfg:
            if value == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(channels, value, 3, padding=1), nn.ReLU(True)]
                channels = value
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def remap_vggface_state(state):
    """Map either torchvision-style or MatConvNet-conversion names."""
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    if any(k.startswith("features.") for k in state):
        return state
    conv_names = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1",
                  "conv3_2", "conv3_3", "conv4_1", "conv4_2", "conv4_3",
                  "conv5_1", "conv5_2", "conv5_3"]
    conv_indexes = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    mapped = {}
    for name, idx in zip(conv_names, conv_indexes):
        for suffix in ("weight", "bias"):
            key = f"{name}.{suffix}"
            if key in state:
                mapped[f"features.{idx}.{suffix}"] = state[key]
    for name, idx in (("fc6", 0), ("fc7", 3), ("fc8", 6)):
        for suffix in ("weight", "bias"):
            key = f"{name}.{suffix}"
            if key in state:
                mapped[f"classifier.{idx}.{suffix}"] = state[key]
    return mapped


def checkpoint_state(path):
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.get("model", obj.get("state_dict", obj))


def build_model(arch, weights_path):
    state = checkpoint_state(weights_path)
    if arch == "facenet":
        from facenet_pytorch import InceptionResnetV1
        # The VGGFace2 checkpoint used a 5,346-identity classification head.
        model = InceptionResnetV1(pretrained=None, classify=True, num_classes=5346)
        model.load_state_dict(state)
        model.logits = nn.Linear(model.logits.in_features, len(CLASSES))
        return model
    if arch in ("resnet18", "resnet50"):
        model = getattr(models, arch)(weights=None, num_classes=5346)
        model.load_state_dict(state)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        return model
    model = VGGFace(2622)
    state = remap_vggface_state(state)
    model.load_state_dict(state)
    model.classifier[6] = nn.Linear(4096, len(CLASSES))
    return model


def make_transforms(arch):
    if arch == "vgg16_vggface":
        # Original VGG-Face RGB mean subtraction on [0,255] pixels.
        mean = [129.1863 / 255, 104.7624 / 255, 93.5940 / 255]
        std = [1 / 255, 1 / 255, 1 / 255]
        size, resize = 224, 256
    else:
        # Matches the RGB preprocessing used for our VGGFace2 pretraining.
        mean, std = [0.5] * 3, [0.5] * 3
        size, resize = 112, 128
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    val = transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize(resize), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    return train, val


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    loss_sum, total = 0.0, 0
    cm = torch.zeros(3, 3, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    for images, target in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, target)
        pred = logits.argmax(1)
        loss_sum += loss.item() * target.numel()
        total += target.numel()
        cm += torch.bincount((target * 3 + pred).cpu(), minlength=9).reshape(3, 3)
    acc = cm.diag().sum().item() / max(total, 1)
    precision = cm.diag() / cm.sum(0).clamp_min(1)
    recall = cm.diag() / cm.sum(1).clamp_min(1)
    macro_f1 = (2 * precision * recall / (precision + recall).clamp_min(1e-12)).mean().item()
    return loss_sum / max(total, 1), acc, macro_f1, cm.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True,
                   choices=["facenet", "vgg16_vggface", "resnet18", "resnet50"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--weights", required=True,
                   help="VGGFace/VGGFace2 pretrained checkpoint for this architecture")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="auto")
    p.add_argument("--max-hours", type=float)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_tf, val_tf = make_transforms(args.arch)
    train_set = datasets.ImageFolder(Path(args.data_root) / "Train", train_tf)
    val_set = datasets.ImageFolder(Path(args.data_root) / "Test", val_tf)
    if train_set.classes != CLASSES or val_set.classes != CLASSES:
        raise ValueError(f"expected classes {CLASSES}; got train={train_set.classes}, test={val_set.classes}")
    common = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, args.batch_size * 2, shuffle=False, **common)
    model = build_model(args.arch, args.weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    start_epoch, best_f1 = 0, -1.0
    resume = out / "last.pt" if args.resume == "auto" else Path(args.resume)
    if resume.is_file():
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_f1 = ckpt["epoch"] + 1, ckpt["best_macro_f1"]
    metrics_path = out / "metrics.csv"
    if start_epoch == 0:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "accuracy", "macro_f1", "lr"])
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, seen = 0.0, 0
        for images, target in train_loader:
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = criterion(model(images), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * target.numel()
            seen += target.numel()
        val_loss, acc, macro_f1, cm = evaluate(model, val_loader, device, amp_dtype)
        scheduler.step()
        state = {"epoch": epoch, "arch": args.arch, "classes": CLASSES,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "best_macro_f1": max(best_f1, macro_f1), "confusion_matrix": cm,
                 "args": vars(args)}
        atomic_save(state, out / "last.pt")
        atomic_save(model.state_dict(), out / "last_weights.pt")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            atomic_save(state, out / "best.pt")
            atomic_save(model.state_dict(), out / "best_weights.pt")
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss / seen, val_loss, acc, macro_f1,
                                    optimizer.param_groups[0]["lr"]])
        print(f"epoch={epoch+1}/{args.epochs} train_loss={train_loss/seen:.4f} "
              f"val_loss={val_loss:.4f} acc={acc:.4f} macro_f1={macro_f1:.4f}", flush=True)
        if args.max_hours and (time.time() - started) / 3600 >= args.max_hours:
            print("stopping cleanly at max-hours", flush=True)
            break


if __name__ == "__main__":
    main()
