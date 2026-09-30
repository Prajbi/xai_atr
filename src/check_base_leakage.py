"""
Check whether the same *root* image (before augmentation) has variants
landing in more than one split. Assumes filenames look like either:
    <basehash>.jpg
    <basehash>_aug_<n>_<m>.jpg

Usage:
    python check_base_leakage.py /path/to/data.yaml
"""

import os
import sys
import glob
import re
from collections import defaultdict

try:
    import yaml
except ImportError:
    print("Install pyyaml first: pip install pyyaml")
    sys.exit(1)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
AUG_PATTERN = re.compile(r"^(.*?)(_aug_\d+_\d+)?$")


def base_id(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = AUG_PATTERN.match(stem)
    return m.group(1) if m else stem


def list_images(images_dir):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    return files


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_base_leakage.py /path/to/data.yaml")
        sys.exit(1)

    yaml_path = sys.argv[1]
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    base_dir = data.get("path", os.path.dirname(os.path.abspath(yaml_path)))

    split_base_ids = {}
    split_file_counts = {}

    for split_key in ("train", "val", "test"):
        if split_key not in data or not data[split_key]:
            continue
        images_dir = os.path.join(base_dir, data[split_key])
        files = list_images(images_dir)
        ids = set(base_id(f) for f in files)
        split_base_ids[split_key] = ids
        split_file_counts[split_key] = len(files)
        print(f"[{split_key}] {len(files)} images, {len(ids)} unique base/root images "
              f"(avg {len(files)/max(1,len(ids)):.1f} augmented variants per root image)")

    print("\n=== Cross-split root-image overlap ===")
    splits = list(split_base_ids.keys())
    total_leak_pct = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = split_base_ids[s1] & split_base_ids[s2]
            pct = 100 * len(overlap) / max(1, len(split_base_ids[s2]))
            print(f"{s1} vs {s2}: {len(overlap)} shared root images "
                  f"(~{pct:.1f}% of {s2}'s root images also appear in {s1})")
            if overlap:
                examples = list(overlap)[:5]
                print(f"  e.g. {examples}")


if __name__ == "__main__":
    main()
