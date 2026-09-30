"""
Check for near-duplicate images leaking across train/val/test splits
using perceptual hashing (catches resized/re-compressed/adjacent-video-frame
duplicates, not just byte-identical files).

Requires: pip install imagehash pillow

Usage:
    python check_leakage.py /path/to/data.yaml --threshold 5
"""

import os
import sys
import glob
import argparse
from collections import defaultdict

try:
    import yaml
except ImportError:
    print("Install pyyaml first: pip install pyyaml")
    sys.exit(1)

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("Install required libs first: pip install imagehash pillow")
    sys.exit(1)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(images_dir):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    return files


def hash_dir(images_dir, sample=None):
    files = list_images(images_dir)
    if sample:
        files = files[:sample]
    hashes = {}
    for i, path in enumerate(files):
        try:
            with Image.open(path) as img:
                h = imagehash.phash(img)
            hashes[path] = h
        except Exception as e:
            print(f"WARNING: could not hash {path}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(files)}")
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path")
    ap.add_argument("--threshold", type=int, default=5,
                     help="Max Hamming distance to count as a near-duplicate (0=exact, ~5=very similar)")
    ap.add_argument("--sample", type=int, default=None,
                     help="Optional: only hash first N images per split (for a quick check on huge datasets)")
    args = ap.parse_args()

    with open(args.yaml_path, "r") as f:
        data = yaml.safe_load(f)
    base = data.get("path", os.path.dirname(os.path.abspath(args.yaml_path)))

    split_dirs = {}
    for split_key in ("train", "val", "test"):
        if split_key in data and data[split_key]:
            split_dirs[split_key] = os.path.join(base, data[split_key])

    all_hashes = {}
    for split_key, img_dir in split_dirs.items():
        print(f"Hashing {split_key} ({img_dir}) ...")
        all_hashes[split_key] = hash_dir(img_dir, sample=args.sample)
        print(f"  -> {len(all_hashes[split_key])} images hashed")

    splits = list(all_hashes.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            print(f"\nComparing {s1} vs {s2} (threshold={args.threshold}) ...")
            matches = 0
            h1_items = list(all_hashes[s1].items())
            h2_items = list(all_hashes[s2].items())
            for p1, h1 in h1_items:
                for p2, h2 in h2_items:
                    if h1 - h2 <= args.threshold:
                        matches += 1
                        if matches <= 10:
                            print(f"  NEAR-DUP: {os.path.basename(p1)}  <->  {os.path.basename(p2)}  (dist={h1 - h2})")
            pct = 100 * matches / max(1, len(h1_items))
            print(f"  Total near-duplicate pairs: {matches} "
                  f"(~{pct:.1f}% of {s1} has a near-dup in {s2})")


if __name__ == "__main__":
    main()
