#!/usr/bin/env python3
"""Create deterministic prefix subsets from the existing VGGFace2 index.

The full index was shuffled once with a fixed seed, so taking its first N rows
gives a reproducible, identity-balanced-enough subset without walking the
715k-file dataset tree again.  Each subset contains exactly N training images.
Validation rows are limited to identities represented in that training subset.
"""

import argparse
from pathlib import Path


def read_rows(path: Path) -> list[tuple[str, int]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                relpath, label = line.rstrip("\n").split("\t")
                rows.append((relpath, int(label)))
    return rows


def write_rows(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w") as handle:
        handle.writelines(f"{relpath}\t{label}\n" for relpath, label in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("index"))
    parser.add_argument("--output-root", type=Path, default=Path("indexes"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[300_000, 500_000])
    args = parser.parse_args()

    train = read_rows(args.source / "train.txt")
    val = read_rows(args.source / "val.txt")
    classes = (args.source / "classes.txt").read_text()

    for size in args.sizes:
        if size <= 0 or size > len(train):
            raise ValueError(f"size must be in [1, {len(train)}], got {size}")

        subset_train = train[:size]
        labels = {label for _, label in subset_train}
        subset_val = [row for row in val if row[1] in labels]
        out = args.output_root / f"{size // 1000}k"
        out.mkdir(parents=True, exist_ok=True)
        (out / "classes.txt").write_text(classes)
        write_rows(out / "train.txt", subset_train)
        write_rows(out / "val.txt", subset_val)
        print(
            f"{out}: train={len(subset_train)} val={len(subset_val)} "
            f"identities={len(labels)}"
        )


if __name__ == "__main__":
    main()
