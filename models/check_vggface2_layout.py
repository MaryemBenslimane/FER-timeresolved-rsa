"""Verify a VGGFace2 copy is usable by train_vggface2_rgb.py -- BEFORE training.

Stdlib only (plus optional Pillow), so it runs without the torch env. Checks the
things that actually break the training run: identity-folder layout, enough
identities for --num-identities, enough images for --max-images within the
FIRST N identities (that is the subset the trainer takes), readable extensions,
and true channel depth of a sample of images.

Usage::

    python check_vggface2_layout.py /path/to/VGGFace2-HQ/train
    python check_vggface2_layout.py /path/to/train --num-identities 4605 --max-images 800000
"""

import argparse
import collections
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--num-identities", type=int, default=4605)
    p.add_argument("--max-images", type=int, default=800_000)
    p.add_argument("--sample", type=int, default=12, help="images to open for a channel check")
    a = p.parse_args()

    if not a.root.is_dir():
        sys.exit(f"FAIL: {a.root} is not a directory")

    subdirs = sorted(d for d in a.root.iterdir() if d.is_dir())
    loose = [f for f in a.root.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT]
    print(f"root                : {a.root}")
    print(f"identity folders    : {len(subdirs)}")
    if loose:
        print(f"  ! {len(loose)} image(s) sit loose in root -- expected one folder per identity")
    if not subdirs:
        sys.exit("FAIL: no identity subdirectories. If this is a nested archive, point at the "
                 "level whose children are identity folders (e.g. .../VGGface2_HQ/train).")

    # a single extra nesting level is the most common packaging surprise
    probe = subdirs[0]
    if not any(f.suffix.lower() in IMG_EXT for f in probe.iterdir() if f.is_file()):
        inner = [d for d in probe.iterdir() if d.is_dir()]
        print(f"  ! {probe.name}/ holds no images directly; it contains {len(inner)} subdir(s)")
        print(f"  ! looks nested one level too deep -- try: {probe}")

    print(f"naming sample       : {[d.name for d in subdirs[:3]]} ... {[d.name for d in subdirs[-2:]]}")

    n_want = a.num_identities or len(subdirs)
    if len(subdirs) < n_want:
        print(f"FAIL: need {n_want} identities, found {len(subdirs)}")
    chosen = subdirs[:n_want]

    counts, exts, empty = [], collections.Counter(), []
    for d in chosen:
        files = [f for f in d.iterdir() if f.is_file()]
        imgs = [f for f in files if f.suffix.lower() in IMG_EXT]
        exts.update(f.suffix.lower() for f in files)
        counts.append(len(imgs))
        if not imgs:
            empty.append(d.name)

    total = sum(counts)
    counts_sorted = sorted(counts)
    print(f"\nfirst {len(chosen)} identities:")
    print(f"  images available  : {total:,}")
    print(f"  per identity      : min {counts_sorted[0]}, median "
          f"{counts_sorted[len(counts_sorted)//2]}, max {counts_sorted[-1]}")
    print(f"  extensions        : {dict(exts)}")
    if empty:
        print(f"  ! {len(empty)} identity folder(s) contain no images: {empty[:5]}")

    cap = a.max_images // max(1, len(chosen))
    short = sum(1 for c in counts if c < cap)
    print(f"\nbudget {a.max_images:,} over {len(chosen)} identities -> cap {cap}/identity")
    print(f"  identities below cap: {short} (their shortfall is redistributed)")

    # channel check -- the trainer forces RGB, but a mostly-grayscale corpus
    # would mean the RGB model is learning from grayscale content
    bad, sampled = [], 0
    try:
        from PIL import Image
        step = max(1, len(chosen) // a.sample)
        modes = collections.Counter()
        sizes = collections.Counter()
        for d in chosen[::step][:a.sample]:
            imgs = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXT]
            if not imgs:
                continue
            try:
                with Image.open(imgs[0]) as im:
                    im.verify()
                with Image.open(imgs[0]) as im:   # verify() leaves the file unusable
                    modes[im.mode] += 1
                    sizes[im.size] += 1
            except Exception as e:                # truncated/corrupt files must not abort the check
                bad.append((imgs[0].name, type(e).__name__))
        sampled = sum(modes.values())
        print(f"\nsampled {sampled} images: modes {dict(modes)}, sizes {dict(sizes)}")
        if bad:
            print(f"  ! {len(bad)} of {len(bad) + sampled} sampled images unreadable: {bad[:3]}")
        if modes and set(modes) <= {"L"}:
            print("  ! every sampled image is single-channel -- this corpus is grayscale")
    except ImportError:
        print("\n(Pillow unavailable -- skipped channel/resolution sampling)")

    ok = (total >= a.max_images and len(subdirs) >= n_want and not empty
          and not bad)
    print(f"\n{'PASS' if ok else 'REVIEW'}: "
          f"{'layout and volume satisfy the requested run' if ok else 'see warnings above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
