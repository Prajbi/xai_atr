import csv
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================

SOURCE = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/aircrafts/J35"
)

OUTPUT = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/yolo_j35_parts"
)

SEED = 42

CLASS_MAP = {
    "J35": 0,
    "nose": 1,
    "wings": 2,
    "tail": 3
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ==========================================
# PROTECT EXISTING DATASET
# ==========================================

if OUTPUT.exists():
    raise SystemExit(
        f"\nERROR: Output directory already exists:\n{OUTPUT}\n"
        "Rename or remove it before running this script again."
    )

# ==========================================
# INDEX IMAGES
# ==========================================

print("\nIndexing images...")

image_index = {}

for p in SOURCE.rglob("*"):
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
        if p.stem in image_index:
            raise SystemExit(f"Duplicate image filename: {p.stem}")

        image_index[p.stem] = p

print("Images indexed:", len(image_index))

# ==========================================
# FIND CSV FILES
# ==========================================

csv_files = list(SOURCE.rglob("*.csv"))

main_files = [
    p for p in csv_files
    if not p.stem.lower().endswith("_parts")
]

part_files = [
    p for p in csv_files
    if p.stem.lower().endswith("_parts")
]

print("Main CSV files:", len(main_files))
print("Parts CSV files:", len(part_files))

# ==========================================
# READ ANNOTATIONS
# ==========================================

main_annotations = defaultdict(list)
part_annotations = defaultdict(list)

def read_annotations(files, is_part):

    for csv_path in files:

        with open(csv_path, "r", newline="") as f:

            reader = csv.DictReader(f)

            for row in reader:

                filename = Path(row["filename"]).stem

                if filename not in image_index:
                    raise ValueError(
                        f"Image missing: {filename}"
                    )

                if is_part:
                    cls = row["part"].strip().lower()

                    if cls not in ["nose", "wings", "tail"]:
                        raise ValueError(
                            f"Unknown component: {cls}"
                        )

                    part_annotations[filename].append(row)

                else:
                    cls = row["class"].strip()

                    if cls != "J35":
                        raise ValueError(
                            f"Unexpected aircraft class: {cls}"
                        )

                    main_annotations[filename].append(row)


read_annotations(main_files, False)
read_annotations(part_files, True)

# ==========================================
# MATCH FULLY ANNOTATED IMAGES
# ==========================================

valid_images = sorted(
    set(main_annotations) & set(part_annotations)
)

excluded_images = (
    set(main_annotations) - set(part_annotations)
)

print("\nFully annotated images:", len(valid_images))
print("Excluded images:", len(excluded_images))

# ==========================================
# VALIDATE AND CONVERT BOXES
# ==========================================

def convert_box(row, class_id, image_width, image_height):

    csv_width = float(row["width"])
    csv_height = float(row["height"])

    if (
        csv_width != image_width
        or csv_height != image_height
    ):
        raise ValueError("Image dimension mismatch")

    xmin = float(row["xmin"])
    ymin = float(row["ymin"])
    xmax = float(row["xmax"])
    ymax = float(row["ymax"])

    if (
        xmin < 0 or ymin < 0
        or xmax > image_width
        or ymax > image_height
        or xmax <= xmin
        or ymax <= ymin
    ):
        raise ValueError("Invalid bounding box")

    x_center = ((xmin + xmax) / 2) / image_width
    y_center = ((ymin + ymax) / 2) / image_height

    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )

# ==========================================
# RANDOM SPLIT
# ==========================================

random.seed(SEED)

random.shuffle(valid_images)

total = len(valid_images)

train_end = int(total * 0.70)
val_end = train_end + int(total * 0.20)

splits = {
    "train": valid_images[:train_end],
    "val": valid_images[train_end:val_end],
    "test": valid_images[val_end:]
}

# ==========================================
# CREATE DIRECTORY STRUCTURE
# ==========================================

for split in splits:

    (OUTPUT / "images" / split).mkdir(
        parents=True, exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

# ==========================================
# EXPORT YOLO DATASET
# ==========================================

class_distribution = {
    split: Counter() for split in splits
}

print("\nConverting annotations...")

for split, filenames in splits.items():

    for filename in filenames:

        image_path = image_index[filename]

        with Image.open(image_path) as img:
            image_width, image_height = img.size

        labels = []

        # Main aircraft annotations
        for row in main_annotations[filename]:

            labels.append(
                convert_box(
                    row, 0,
                    image_width, image_height
                )
            )

            class_distribution[split]["J35"] += 1

        # Component annotations
        for row in part_annotations[filename]:

            component = row["part"].strip().lower()
            class_id = CLASS_MAP[component]

            labels.append(
                convert_box(
                    row, class_id,
                    image_width, image_height
                )
            )

            class_distribution[split][component] += 1

        # Copy image
        shutil.copy2(
            image_path,
            OUTPUT / "images" / split / image_path.name
        )

        # Write combined YOLO labels
        label_path = (
            OUTPUT / "labels" / split / f"{filename}.txt"
        )

        with open(label_path, "w") as f:
            f.write("\n".join(labels) + "\n")

# ==========================================
# CREATE DATA.YAML
# ==========================================

yaml_content = f"""path: {OUTPUT}

train: images/train
val: images/val
test: images/test

nc: 4

names:
  0: J35
  1: nose
  2: wings
  3: tail
"""

with open(OUTPUT / "data.yaml", "w") as f:
    f.write(yaml_content)

# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 55)
print("J35 YOLO DATASET CREATED SUCCESSFULLY")
print("=" * 55)

for split, filenames in splits.items():

    print(f"\n{split.upper()}")
    print("Images:", len(filenames))

    for cls in ["J35", "nose", "wings", "tail"]:
        print(
            f"{cls}: {class_distribution[split][cls]}"
        )

print("\nTOTAL IMAGES:", total)
print("EXCLUDED IMAGES:", len(excluded_images))

print("\nOUTPUT DIRECTORY:")
print(OUTPUT)

print("\nDATA YAML:")
print(OUTPUT / "data.yaml")

print("\nOriginal dataset was not modified.")
print("=" * 55)
