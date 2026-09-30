"""
For the train split, report per class: total labeled images, unique root
images (before augmentation), and average augmented variants per root
image. Flags classes that look like they're mostly synthetic repeats of
a small number of real photos.

Usage:
    python check_class_diversity.py /path/to/data.yaml
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

AUG_PATTERN = re.compile(r"^(.*?)(_aug_\d+_\d+)?$")


def base_id(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = AUG_PATTERN.match(stem)
    return m.group(1) if m else stem


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_class_diversity.py /path/to/data.yaml")
        sys.exit(1)

    yaml_path = sys.argv[1]
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    base_dir = data.get("path", os.path.dirname(os.path.abspath(yaml_path)))
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    nc = len(names)

    train_images_dir = os.path.join(base_dir, data["train"])
    train_labels_dir = train_images_dir.replace("images", "labels")

    label_files = glob.glob(os.path.join(train_labels_dir, "*.txt"))
    print(f"Scanning {len(label_files)} train label files...")

    # class -> set of root ids, class -> total image count
    class_roots = defaultdict(set)
    class_images = defaultdict(int)

    for lf in label_files:
        rid = base_id(lf)
        with open(lf, "r") as f:
            classes_here = set()
            for line in f:
                line = line.strip()
                if line:
                    classes_here.add(int(line.split()[0]))
        for c in classes_here:
            class_roots[c].add(rid)
            class_images[c] += 1

    rows = []
    for c in range(nc):
        n_images = class_images.get(c, 0)
        n_roots = len(class_roots.get(c, []))
        ratio = n_images / n_roots if n_roots else 0
        rows.append((names[c], n_images, n_roots, ratio))

    # Sort by root-image count ascending (the real bottleneck signal)
    rows.sort(key=lambda r: r[2])

    print(f"\n{'Class':<15}{'Images':>10}{'RootImgs':>12}{'Variants/Root':>16}")
    print("-" * 55)
    for name, n_images, n_roots, ratio in rows:
        flag = "  <-- LOW ROOT DIVERSITY" if n_roots < 100 else ""
        print(f"{name:<15}{n_images:>10}{n_roots:>12}{ratio:>16.1f}{flag}")

    low = [r for r in rows if r[2] < 100]
    print(f"\n{len(low)} / {nc} classes have fewer than 100 unique root images.")
    if low:
        print("Lowest root-diversity classes:", ", ".join(r[0] for r in low[:20]))


if __name__ == "__main__":
    main()
