from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/home/admin1/Abishek-Intern/new_atr/data"
)

# ORIGINAL DATASET
ORIGINAL_DIR = (
    BASE / "raw" / "dataset"
)

# BALANCED DATASET
BALANCED_DIR = (
    BASE / "raw" / "balanced_dataset"
)

BALANCED_CSV = (
    BALANCED_DIR /
    "labels_with_split_balanced.csv"
)

AUGMENTED_DIR = (
    BALANCED_DIR / "images"
)

# FINAL YOLO DATASET
OUTPUT = (
    BASE / "yolo_aircraft"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

print("=" * 70)
print("CREATING YOLO DATASET")
print("=" * 70)

for split in ["train", "val", "test"]:

    (
        OUTPUT /
        "images" /
        split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT /
        "labels" /
        split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD BALANCED CSV
# ============================================================

print("\nLoading balanced CSV...")

df = pd.read_csv(
    BALANCED_CSV
)

df["filename"] = (
    df["filename"]
    .astype(str)
)

df["class"] = (
    df["class"]
    .astype(str)
)

df["split"] = (
    df["split"]
    .astype(str)
    .str.lower()
)


print(
    f"Annotations       : {len(df):,}"
)

print(
    f"Unique images     : "
    f"{df['filename'].nunique():,}"
)

print(
    f"Classes           : "
    f"{df['class'].nunique()}"
)


# ============================================================
# CLASS MAPPING
# ============================================================

classes = sorted(
    df["class"].unique()
)

class_to_id = {
    name: idx
    for idx, name in enumerate(classes)
}


print("\nNumber of classes:", len(classes))

print("\nClass mapping:")

for name, idx in class_to_id.items():

    print(
        f"{idx:3d} : {name}"
    )


# ============================================================
# SAVE CLASSES
# ============================================================

with open(
    OUTPUT / "classes.txt",
    "w"
) as f:

    for name in classes:

        f.write(
            name + "\n"
        )


# ============================================================
# IMAGE LOOKUP
# ============================================================

def find_original_image(filename):

    """
    Find an original image inside raw/dataset.
    """

    direct = (
        ORIGINAL_DIR / filename
    )

    if direct.exists():
        return direct


    # CSV normally stores filename without extension
    for ext in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ]:

        candidate = (
            ORIGINAL_DIR /
            f"{filename}{ext}"
        )

        if candidate.exists():
            return candidate


    # Stem fallback
    matches = list(
        ORIGINAL_DIR.glob(
            f"{filename}.*"
        )
    )

    for path in matches:

        if path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }:

            return path


    return None


def find_augmented_image(filename):

    """
    Find an augmented image inside
    raw/balanced_dataset/images.
    """

    direct = (
        AUGMENTED_DIR / filename
    )

    if direct.exists():
        return direct


    for ext in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ]:

        candidate = (
            AUGMENTED_DIR /
            f"{filename}{ext}"
        )

        if candidate.exists():
            return candidate


    return None


# ============================================================
# GROUP ANNOTATIONS
# ============================================================

groups = (
    df.groupby(
        ["split", "filename"]
    )
)


# ============================================================
# COUNTERS
# ============================================================

processed = {
    "train": 0,
    "val": 0,
    "test": 0
}

missing = []

annotations_written = 0


# ============================================================
# CONVERSION
# ============================================================

print()
print("=" * 70)
print("CONVERTING")
print("=" * 70)

for (
    split,
    filename
), group in tqdm(
    groups,
    total=len(groups),
    desc="Images"
):

    # --------------------------------------------------------
    # Validate split
    # --------------------------------------------------------

    if split not in [
        "train",
        "val",
        "test"
    ]:

        continue


    # --------------------------------------------------------
    # Determine whether this is an augmented image
    # --------------------------------------------------------

    is_augmented = (
        "_aug_" in filename
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Original images come from:
    #     raw/dataset
    #
    # Augmented images come from:
    #     raw/balanced_dataset/images
    # --------------------------------------------------------

    if is_augmented:

        source_image = (
            find_augmented_image(
                filename
            )
        )

    else:

        source_image = (
            find_original_image(
                filename
            )
        )


    if source_image is None:

        missing.append(
            (
                split,
                filename
            )
        )

        continue


    # --------------------------------------------------------
    # Destination image
    # --------------------------------------------------------

    destination_image = (
        OUTPUT /
        "images" /
        split /
        source_image.name
    )


    # Copy only if not already there
    if not destination_image.exists():

        shutil.copy2(
            source_image,
            destination_image
        )


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    width = float(
        group.iloc[0]["width"]
    )

    height = float(
        group.iloc[0]["height"]
    )


    if width <= 0 or height <= 0:

        print(
            f"\nWARNING: Invalid dimensions: "
            f"{filename}"
        )

        continue


    # --------------------------------------------------------
    # YOLO label path
    # --------------------------------------------------------

    label_path = (
        OUTPUT /
        "labels" /
        split /
        f"{source_image.stem}.txt"
    )


    lines = []


    # --------------------------------------------------------
    # Convert bounding boxes
    # --------------------------------------------------------

    for _, row in group.iterrows():

        class_name = str(
            row["class"]
        )

        class_id = (
            class_to_id[class_name]
        )


        xmin = float(
            row["xmin"]
        )

        ymin = float(
            row["ymin"]
        )

        xmax = float(
            row["xmax"]
        )

        ymax = float(
            row["ymax"]
        )


        # ----------------------------------------------------
        # Clamp
        # ----------------------------------------------------

        xmin = max(
            0.0,
            min(xmin, width)
        )

        ymin = max(
            0.0,
            min(ymin, height)
        )

        xmax = max(
            0.0,
            min(xmax, width)
        )

        ymax = max(
            0.0,
            min(ymax, height)
        )


        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if xmax <= xmin:
            continue

        if ymax <= ymin:
            continue


        # ----------------------------------------------------
        # Pascal VOC -> YOLO
        # ----------------------------------------------------

        x_center = (
            (xmin + xmax) / 2
        )

        y_center = (
            (ymin + ymax) / 2
        )

        box_width = (
            xmax - xmin
        )

        box_height = (
            ymax - ymin
        )


        # Normalize
        x_center /= width
        y_center /= height
        box_width /= width
        box_height /= height


        # ----------------------------------------------------
        # Final safety clamp
        # ----------------------------------------------------

        x_center = max(
            0.0,
            min(x_center, 1.0)
        )

        y_center = max(
            0.0,
            min(y_center, 1.0)
        )

        box_width = max(
            0.0,
            min(box_width, 1.0)
        )

        box_height = max(
            0.0,
            min(box_height, 1.0)
        )


        lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        annotations_written += 1


    # --------------------------------------------------------
    # Write label file
    # --------------------------------------------------------

    with open(
        label_path,
        "w"
    ) as f:

        f.write(
            "\n".join(lines)
        )


    processed[split] += 1


# ============================================================
# CREATE data.yaml
# ============================================================

yaml_path = (
    OUTPUT / "data.yaml"
)


with open(
    yaml_path,
    "w"
) as f:

    f.write(
        f"path: {OUTPUT.resolve()}\n"
    )

    f.write(
        "train: images/train\n"
    )

    f.write(
        "val: images/val\n"
    )

    f.write(
        "test: images/test\n"
    )

    f.write("\n")

    f.write(
        f"nc: {len(classes)}\n"
    )

    f.write(
        "names:\n"
    )

    for idx, name in enumerate(
        classes
    ):

        # Escape single quotes
        safe_name = (
            name.replace(
                "'",
                "''"
            )
        )

        f.write(
            f"  {idx}: '{safe_name}'\n"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("YOLO CONVERSION COMPLETE")
print("=" * 70)

print(
    f"Train images : "
    f"{processed['train']:,}"
)

print(
    f"Val images   : "
    f"{processed['val']:,}"
)

print(
    f"Test images  : "
    f"{processed['test']:,}"
)

print(
    f"Annotations  : "
    f"{annotations_written:,}"
)

print(
    f"Missing      : "
    f"{len(missing):,}"
)

print()
print(
    f"Output:\n{OUTPUT}"
)

print()
print(
    f"data.yaml:\n{yaml_path}"
)


if missing:

    print()
    print("=" * 70)
    print("MISSING IMAGES")
    print("=" * 70)

    for split, filename in missing[:30]:

        print(
            split,
            filename
        )
