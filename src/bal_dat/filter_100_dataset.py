import os
import shutil
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/raw/labels_with_split.csv"
)

SOURCE_DATASET = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/raw/dataset"
)

OUTPUT_DATASET = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/filtered_100"
)

MIN_IMAGES = 100


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# LOAD CSV
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(CSV_PATH)

required_columns = {
    "filename",
    "width",
    "height",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "split",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

print(f"Total annotations : {len(df):,}")
print(f"Total classes     : {df['class'].nunique()}")
print(f"Splits            : {df['split'].unique().tolist()}")


# ============================================================
# COUNT UNIQUE IMAGES PER CLASS
# ============================================================

print()
print("=" * 70)
print("COUNTING UNIQUE IMAGES PER CLASS")
print("=" * 70)

image_count = (
    df.groupby("class")["filename"]
    .nunique()
    .sort_values(ascending=False)
)

selected_classes = image_count[image_count >= MIN_IMAGES].index.tolist()

print(f"Minimum images required : {MIN_IMAGES}")
print(f"Classes with >= {MIN_IMAGES} images : {len(selected_classes)}")

print()
print("SELECTED CLASSES")
print("-" * 70)

for cls in selected_classes:
    print(f"{cls:<20} {image_count[cls]:>6} images")


# ============================================================
# CLASS MAPPING
# ============================================================

selected_classes = sorted(selected_classes)

class_to_id = {
    cls: idx
    for idx, cls in enumerate(selected_classes)
}

print()
print("=" * 70)
print("YOLO CLASS MAPPING")
print("=" * 70)

for cls, idx in class_to_id.items():
    print(f"{idx:>3} -> {cls}")


# ============================================================
# FILTER ANNOTATIONS
# ============================================================

filtered_df = df[df["class"].isin(selected_classes)].copy()

print()
print("=" * 70)
print("FILTERED DATA")
print("=" * 70)

print(f"Original annotations : {len(df):,}")
print(f"Filtered annotations : {len(filtered_df):,}")

print(
    f"Original images      : {df['filename'].nunique():,}"
)

print(
    f"Filtered images      : {filtered_df['filename'].nunique():,}"
)


# ============================================================
# FIND SOURCE IMAGES
# ============================================================

print()
print("=" * 70)
print("INDEXING SOURCE IMAGES")
print("=" * 70)

source_images = {}

for path in SOURCE_DATASET.rglob("*"):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        source_images[path.stem] = path

print(f"Images found in source dataset: {len(source_images):,}")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "validation", "test"]:
    (OUTPUT_DATASET / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DATASET / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CREATE YOLO LABELS
# ============================================================

print()
print("=" * 70)
print("CREATING FILTERED YOLO DATASET")
print("=" * 70)

processed_images = set()
missing_images = []

grouped = filtered_df.groupby(
    ["split", "filename"],
    sort=False
)

for (split, filename), group in grouped:

    if split not in ["train", "validation", "test"]:
        print(
            f"WARNING: Unknown split '{split}' "
            f"for {filename}"
        )
        continue

    # --------------------------------------------------------
    # Find source image
    # --------------------------------------------------------

    source_image = source_images.get(str(filename))

    if source_image is None:

        # Try filename with possible extension
        candidates = []

        for ext in IMAGE_EXTENSIONS:
            candidate = SOURCE_DATASET / f"{filename}{ext}"

            if candidate.exists():
                candidates.append(candidate)

        if candidates:
            source_image = candidates[0]

    if source_image is None:
        missing_images.append(filename)
        continue


    # --------------------------------------------------------
    # Output image
    # --------------------------------------------------------

    output_image = (
        OUTPUT_DATASET
        / "images"
        / split
        / source_image.name
    )

    shutil.copy2(
        source_image,
        output_image
    )


    # --------------------------------------------------------
    # YOLO label
    # --------------------------------------------------------

    label_lines = []

    for _, row in group.iterrows():

        class_name = row["class"]
        class_id = class_to_id[class_name]

        width = float(row["width"])
        height = float(row["height"])

        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])

        # Bounding-box center
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0

        # Bounding-box dimensions
        box_width = xmax - xmin
        box_height = ymax - ymin

        # Normalize
        x_center /= width
        y_center /= height
        box_width /= width
        box_height /= height

        # Clamp values
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        box_width = max(0.0, min(1.0, box_width))
        box_height = max(0.0, min(1.0, box_height))

        label_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )


    # --------------------------------------------------------
    # Save label
    # --------------------------------------------------------

    label_file = (
        OUTPUT_DATASET
        / "labels"
        / split
        / f"{source_image.stem}.txt"
    )

    with open(label_file, "w") as f:
        f.write("\n".join(label_lines))

    processed_images.add(filename)


# ============================================================
# DATASET YAML
# ============================================================

yaml_path = OUTPUT_DATASET / "data.yaml"

with open(yaml_path, "w") as f:

    f.write(f"path: {OUTPUT_DATASET}\n")
    f.write("train: images/train\n")
    f.write("val: images/validation\n")
    f.write("test: images/test\n")

    f.write("\n")

    f.write(f"nc: {len(selected_classes)}\n")

    f.write("names:\n")

    for idx, cls in enumerate(selected_classes):
        f.write(f"  {idx}: {cls}\n")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("DATASET CREATION COMPLETE")
print("=" * 70)

print(f"Output dataset : {OUTPUT_DATASET}")
print(f"Classes        : {len(selected_classes)}")
print(f"Images copied  : {len(processed_images):,}")
print(f"Missing images : {len(set(missing_images)):,}")

print()
print("Split summary:")
print("-" * 70)

for split in ["train", "validation", "test"]:

    image_dir = OUTPUT_DATASET / "images" / split
    label_dir = OUTPUT_DATASET / "labels" / split

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    print(
        f"{split:<12} "
        f"images={len(images):>6} "
        f"labels={len(labels):>6}"
    )

print()
print(f"YAML file: {yaml_path}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)
