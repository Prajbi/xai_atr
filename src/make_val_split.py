"""
Create a stratified val split by moving a fraction of images (and their
labels) out of train/ into val/, ensuring every class gets a fair share
even if it's rare overall.

Usage:
    python make_val_split.py /path/to/data.yaml --val-frac 0.12 --min-val 10
"""

import sys
import os
import glob
import random
import shutil
import argparse
from collections import defaultdict

try:
    import yaml
except ImportError:
    print("Install pyyaml first: pip install pyyaml")
    sys.exit(1)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_image_for_label(label_path, images_dir):
    stem = os.path.splitext(os.path.basename(label_path))[0]
    for ext in IMG_EXTS:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path")
    ap.add_argument("--val-frac", type=float, default=0.12,
                     help="Target fraction of each class's images to move to val")
    ap.add_argument("--min-val", type=int, default=8,
                     help="Minimum images per class in val, if the class has enough")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                     help="Print planned counts without moving any files")
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.yaml_path, "r") as f:
        data = yaml.safe_load(f)

    base = data.get("path", os.path.dirname(os.path.abspath(args.yaml_path)))
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    nc = len(names)

    train_images_dir = os.path.join(base, data["train"])
    train_labels_dir = train_images_dir.replace("images", "labels")
    val_images_dir = os.path.join(base, data.get("val", "images/val"))
    val_labels_dir = val_images_dir.replace("images", "labels")

    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    label_files = glob.glob(os.path.join(train_labels_dir, "*.txt"))
    print(f"Found {len(label_files)} train label files.")

    # image -> set of classes present
    image_classes = {}
    class_to_images = defaultdict(set)
    for lf in label_files:
        with open(lf, "r") as f:
            classes_here = set()
            for line in f:
                line = line.strip()
                if line:
                    classes_here.add(int(line.split()[0]))
        image_classes[lf] = classes_here
        for c in classes_here:
            class_to_images[c].add(lf)

    # Process rarest classes first so they get priority for val allocation
    class_order = sorted(range(nc), key=lambda c: len(class_to_images.get(c, [])))

    assigned = {}  # label_file -> 'val' or 'train'
    val_count_per_class = defaultdict(int)
    total_count_per_class = {c: len(class_to_images.get(c, [])) for c in range(nc)}

    for c in class_order:
        imgs = list(class_to_images.get(c, []))
        random.shuffle(imgs)
        target = max(args.min_val, round(len(imgs) * args.val_frac)) if imgs else 0
        target = min(target, len(imgs))

        for lf in imgs:
            if lf in assigned:
                # already decided by a rarer class; count it if it landed in val
                if assigned[lf] == "val":
                    val_count_per_class[c] += 1
                continue
            if val_count_per_class[c] < target:
                assigned[lf] = "val"
                val_count_per_class[c] += 1
                for cc in image_classes[lf]:
                    if cc != c:
                        val_count_per_class[cc] += 1  # will be recounted correctly below
            else:
                assigned[lf] = "train"

    # Recompute accurate per-class val counts from final assignment (avoids double count drift)
    val_count_per_class = defaultdict(int)
    for lf, split in assigned.items():
        if split == "val":
            for c in image_classes[lf]:
                val_count_per_class[c] += 1

    print(f"\n{'Class':<15}{'Total':>8}{'Val (planned)':>15}")
    print("-" * 38)
    for c in range(nc):
        print(f"{names[c]:<15}{total_count_per_class[c]:>8}{val_count_per_class[c]:>15}")

    n_val = sum(1 for s in assigned.values() if s == "val")
    n_train = sum(1 for s in assigned.values() if s == "train")
    print(f"\nPlanned: {n_train} stay in train, {n_val} move to val.")

    if args.dry_run:
        print("\nDry run only — no files moved. Re-run without --dry-run to apply.")
        return

    moved = 0
    for lf, split in assigned.items():
        if split != "val":
            continue
        img_path = find_image_for_label(lf, train_images_dir)
        if img_path is None:
            print(f"WARNING: no image found for label {lf}, skipping.")
            continue
        shutil.move(lf, os.path.join(val_labels_dir, os.path.basename(lf)))
        shutil.move(img_path, os.path.join(val_images_dir, os.path.basename(img_path)))
        moved += 1

    print(f"\nMoved {moved} image/label pairs into val.")


if __name__ == "__main__":
    main()
