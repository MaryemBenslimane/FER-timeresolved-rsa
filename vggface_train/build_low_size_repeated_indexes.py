#!/usr/bin/env python3
"""Build ten independent 10k/50k/100k/200k VGGFace2 subset indexes."""
import json
import random
from pathlib import Path

ROOT = Path("./vggface_train")
SOURCE = ROOT / "index"
OUT = ROOT / "indexes_repeated"
SIZES = {"10k": 10_000, "50k": 50_000, "100k": 100_000, "200k": 200_000}


def rows(path):
    result = []
    for line in path.read_text().splitlines():
        if line:
            name, label = line.split("\t")
            result.append((name, int(label)))
    return result


def write(path, data):
    path.write_text("".join(f"{name}\t{label}\n" for name, label in data))


train, val = rows(SOURCE / "train.txt"), rows(SOURCE / "val.txt")
classes = (SOURCE / "classes.txt").read_text()
for repeat in range(1, 11):
    seed = 1000 + repeat
    shuffled = train.copy()
    random.Random(seed).shuffle(shuffled)
    for condition, size in SIZES.items():
        selected = shuffled[:size]
        labels = {label for _, label in selected}
        selected_val = [row for row in val if row[1] in labels]
        dest = OUT / f"repeat_{repeat:02d}" / condition
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "classes.txt").write_text(classes)
        write(dest / "train.txt", selected)
        write(dest / "val.txt", selected_val)
        (dest / "sampling.json").write_text(json.dumps({
            "repeat": repeat, "seed": seed, "condition": condition,
            "sampling": "without_replacement", "source_pool_images": len(train),
            "training_images": size, "unique_training_images": size,
            "represented_identities": len(labels), "validation_images": len(selected_val),
        }, indent=2))
        print(f"{dest}: train={size} identities={len(labels)} val={len(selected_val)}")
