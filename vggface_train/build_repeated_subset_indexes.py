#!/usr/bin/env python3
"""Build 10 independent FaceNet subset indexes for the pretraining-size study."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_rows(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                relpath, label = line.rstrip("\n").split("\t")
                rows.append((relpath, int(label)))
    return rows


def write_rows(path, rows):
    with Path(path).open("w") as handle:
        handle.writelines(f"{path}\t{label}\n" for path, label in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("vggface_train/index"))
    ap.add_argument("--out", type=Path, default=Path("vggface_train/indexes_repeated"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1001)
    args = ap.parse_args()

    train = read_rows(args.source / "train.txt")
    val = read_rows(args.source / "val.txt")
    classes = (args.source / "classes.txt").read_text()
    if len(train) != 715_810:
        raise RuntimeError(f"expected the audited 715,810-image pool, got {len(train)}")

    for repeat in range(1, args.repeats + 1):
        seed = args.seed_start + repeat - 1
        rng = random.Random(seed)
        shuffled = train.copy()
        rng.shuffle(shuffled)
        selections = {
            "300k": shuffled[:300_000],
            "500k": shuffled[:500_000],
            # Bootstrap exposure: 800k draws from the 715,810 unique-image pool.
            "800k_bootstrap": [train[rng.randrange(len(train))] for _ in range(800_000)],
        }
        for size, selected in selections.items():
            labels = {label for _, label in selected}
            selected_val = [row for row in val if row[1] in labels]
            dest = args.out / f"repeat_{repeat:02d}" / size
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "classes.txt").write_text(classes)
            write_rows(dest / "train.txt", selected)
            write_rows(dest / "val.txt", selected_val)
            counts = Counter(path for path, _ in selected)
            metadata = {
                "repeat": repeat, "seed": seed, "condition": size,
                "sampling": "with_replacement" if size == "800k_bootstrap" else "without_replacement",
                "source_pool_images": len(train), "training_draws": len(selected),
                "unique_training_images": len(counts), "duplicate_draws": len(selected) - len(counts),
                "represented_identities": len(labels), "validation_images": len(selected_val),
            }
            (dest / "sampling.json").write_text(json.dumps(metadata, indent=2))
            print(dest, metadata)


if __name__ == "__main__":
    main()
