import os
import cv2
import random
import shutil
import hashlib
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
import albumentations as A


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/home/admin1/Abishek-Intern/new_atr/data/raw")

CSV_PATH = BASE_DIR / "labels_with_split.csv"

# Your possible source directories.
SOURCE_DIRS = [
    BASE_DIR / "raw" / "crop",
    BASE_DIR / "dataset",
    BASE_DIR / "annotated_samples",
]

# New dataset -- original data will NOT be modified.
OUTPUT_DIR = BASE_DIR / "balanced_dataset"

IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"

OUTPUT_CSV = OUTPUT_DIR / "labels_with_split_balanced.csv"

DISTRIBUTION_CSV = OUTPUT_DIR / "class_distribution_before_after.csv"


# Classes below this number will be augmented.
TARGET_IMAGES_PER_CLASS = 150

# Random seed for reproducibility.
SEED = 42

# Maximum number of augmentation attempts for one class.
MAX_ATTEMPTS_MULTIPLIER = 20


random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# AUGMENTATION PIPELINE
# ============================================================

# Bboxes are in Pascal VOC format:
# [xmin, ymin, xmax, ymax]

AUGMENTATION = A.Compose(
    [
        A.HorizontalFlip(p=0.50),

        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(-0.08, 0.08),
            rotate=(-8, 8),
            shear=(-3, 3),
            p=0.75,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.20,
                    contrast_limit=0.20,
                    p=1.0,
                ),

                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=1.0,
                ),
            ],
            p=0.40,
        ),

        A.OneOf(
            [
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    p=1.0,
                ),

                A.GaussNoise(
                    p=1.0,
                ),
            ],
            p=0.15,
        ),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=0.30,
        min_area=4,
    ),
)


# ============================================================
# HELPERS
# ============================================================

def find_image(filename):
    """
    Find an image by filename.

    CSV may contain:
        abc123

    while actual file may be:
        abc123.jpg
        abc123.jpeg
        abc123.png
        etc.
    """

    filename = str(filename)

    # If CSV already contains extension.
    possible_extensions = [
        "",
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]

    for source_dir in SOURCE_DIRS:

        if not source_dir.exists():
            continue

        for ext in possible_extensions:

            candidate = source_dir / (filename + ext)

            if candidate.exists():
                return candidate

    # Recursive search as fallback.
    for source_dir in SOURCE_DIRS:

        if not source_dir.exists():
            continue

        for ext in possible_extensions:

            matches = list(source_dir.rglob(filename + ext))

            if matches:
                return matches[0]

    return None


def make_unique_filename(original_filename, augmentation_id):
    """
    Generate a deterministic unique filename.
    """

    stem = Path(str(original_filename)).stem

    hash_input = f"{stem}_{augmentation_id}_{SEED}"

    short_hash = hashlib.md5(
        hash_input.encode()
    ).hexdigest()[:10]

    return f"{stem}_aug_{augmentation_id}_{short_hash}.jpg"


def clamp_bbox(xmin, ymin, xmax, ymax, width, height):

    xmin = max(0, min(float(xmin), width - 1))
    ymin = max(0, min(float(ymin), height - 1))

    xmax = max(0, min(float(xmax), width - 1))
    ymax = max(0, min(float(ymax), height - 1))

    return xmin, ymin, xmax, ymax


def valid_bbox(bbox, width, height):

    xmin, ymin, xmax, ymax = bbox

    xmin, ymin, xmax, ymax = clamp_bbox(
        xmin,
        ymin,
        xmax,
        ymax,
        width,
        height,
    )

    if xmax <= xmin:
        return False

    if ymax <= ymin:
        return False

    return True


# ============================================================
# LOAD CSV
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(CSV_PATH)

required_columns = [
    "filename",
    "width",
    "height",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "split",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in CSV: {missing}"
    )

print(f"CSV: {CSV_PATH}")
print(f"Total annotations: {len(df):,}")
print(f"Classes: {df['class'].nunique()}")
print()
print("Split distribution:")
print(df["split"].value_counts())
print()


# ============================================================
# PREPARE OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# COPY ORIGINAL TRAIN IMAGES
# ============================================================

print("=" * 70)
print("LOCATING ORIGINAL IMAGES")
print("=" * 70)

all_filenames = df["filename"].astype(str).unique()

image_cache = {}

missing_images = []

for filename in tqdm(
    all_filenames,
    desc="Finding images",
):

    image_path = find_image(filename)

    if image_path is None:
        missing_images.append(filename)
    else:
        image_cache[filename] = image_path


print()
print(f"Images referenced by CSV: {len(all_filenames):,}")
print(f"Images found: {len(image_cache):,}")
print(f"Images missing: {len(missing_images):,}")


if missing_images:

    print("\nFirst 20 missing images:")

    for x in missing_images[:20]:
        print("  ", x)


# ============================================================
# COPY ORIGINAL DATA
# ============================================================

print()
print("=" * 70)
print("COPYING ORIGINAL IMAGES")
print("=" * 70)

original_filenames = set()

for filename, image_path in tqdm(
    image_cache.items(),
    desc="Copying images",
):

    destination = IMAGE_OUTPUT_DIR / Path(image_path).name

    # If duplicate names occur, preserve only one physical copy.
    if not destination.exists():

        shutil.copy2(
            image_path,
            destination,
        )

    original_filenames.add(filename)


# ============================================================
# TRAIN DATA ONLY
# ============================================================

train_df = df[
    df["split"].astype(str).str.lower() == "train"
].copy()

print()
print("=" * 70)
print("TRAINING DATA")
print("=" * 70)

train_unique_images = (
    train_df.groupby("class")["filename"]
    .nunique()
    .sort_values()
)

print(
    f"Unique training images: "
    f"{train_df['filename'].nunique():,}"
)

print(
    f"Training classes: "
    f"{train_df['class'].nunique()}"
)


# ============================================================
# FIND MINORITY CLASSES
# ============================================================

minority_classes = train_unique_images[
    train_unique_images < TARGET_IMAGES_PER_CLASS
]

print()
print("=" * 70)
print("MINORITY CLASSES")
print("=" * 70)

print(
    f"Target images/class: "
    f"{TARGET_IMAGES_PER_CLASS}"
)

print(
    f"Minority classes: "
    f"{len(minority_classes)}"
)

print()

for cls, count in minority_classes.items():

    print(
        f"{str(cls):25s} "
        f"{count:5d} -> "
        f"{TARGET_IMAGES_PER_CLASS}"
    )


# ============================================================
# ORIGINAL CLASS DISTRIBUTION
# ============================================================

before_distribution = (
    train_unique_images
    .rename("before_images")
    .reset_index()
)

before_distribution.columns = [
    "class",
    "before_images",
]


# ============================================================
# CREATE GROUPED IMAGE RECORDS
# ============================================================

# Each image can contain multiple bounding boxes/classes.
#
# We therefore group by filename and keep ALL annotations.
#
# This is critical:
#
# If an image contains:
#
#     F16
#     F16
#     MiG29
#
# and we augment that image,
#
# all three bounding boxes are transformed together.


image_groups = {}

for filename, group in train_df.groupby("filename"):

    image_groups[str(filename)] = group.copy()


# ============================================================
# AUGMENT MINORITY CLASSES
# ============================================================

augmented_rows = []

augmentation_counter = 0

print()
print("=" * 70)
print("STARTING TARGETED AUGMENTATION")
print("=" * 70)


for target_class, current_count in minority_classes.items():

    target_class = str(target_class)

    required = (
        TARGET_IMAGES_PER_CLASS
        - int(current_count)
    )

    if required <= 0:
        continue

    print()
    print("-" * 70)
    print(
        f"Class: {target_class}"
    )

    print(
        f"Current unique images: {current_count}"
    )

    print(
        f"Images required: {required}"
    )

    # --------------------------------------------------------
    # Candidate images containing target class
    # --------------------------------------------------------

    candidate_filenames = []

    for filename, group in image_groups.items():

        classes_in_image = (
            group["class"]
            .astype(str)
            .tolist()
        )

        if target_class in classes_in_image:
            candidate_filenames.append(filename)

    if not candidate_filenames:

        print(
            f"WARNING: No source images found "
            f"for {target_class}"
        )

        continue

    print(
        f"Candidate source images: "
        f"{len(candidate_filenames)}"
    )

    generated_for_class = 0
    attempts = 0

    max_attempts = max(
        required * MAX_ATTEMPTS_MULTIPLIER,
        100,
    )

    # --------------------------------------------------------
    # Generate augmentations
    # --------------------------------------------------------

    progress = tqdm(
        total=required,
        desc=f"Augmenting {target_class}",
    )

    while (
        generated_for_class < required
        and attempts < max_attempts
    ):

        attempts += 1

        source_filename = random.choice(
            candidate_filenames
        )

        source_path = image_cache.get(
            str(source_filename)
        )

        if source_path is None:
            continue

        image = cv2.imread(
            str(source_path)
        )

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        group = image_groups[
            str(source_filename)
        ]

        bboxes = []
        labels = []

        for _, row in group.iterrows():

            bbox = [
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            ]

            if not valid_bbox(
                bbox,
                image.shape[1],
                image.shape[0],
            ):
                continue

            bboxes.append(bbox)

            labels.append(
                str(row["class"])
            )

        if not bboxes:
            continue

        # ----------------------------------------------------
        # Apply augmentation
        # ----------------------------------------------------

        try:

            result = AUGMENTATION(
                image=image,
                bboxes=bboxes,
                class_labels=labels,
            )

        except Exception:
            continue

        augmented_image = result["image"]
        transformed_bboxes = result["bboxes"]
        transformed_labels = result[
            "class_labels"
        ]

        if len(transformed_bboxes) == 0:
            continue

        # ----------------------------------------------------
        # Make sure target class survived
        # ----------------------------------------------------

        target_exists = any(
            str(label) == target_class
            for label in transformed_labels
        )

        if not target_exists:
            continue

        new_height, new_width = (
            augmented_image.shape[:2]
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        augmentation_counter += 1

        new_filename = make_unique_filename(
            source_filename,
            augmentation_counter,
        )

        output_image_path = (
            IMAGE_OUTPUT_DIR /
            new_filename
        )

        bgr_image = cv2.cvtColor(
            augmented_image,
            cv2.COLOR_RGB2BGR,
        )

        success = cv2.imwrite(
            str(output_image_path),
            bgr_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not success:
            continue

        # ----------------------------------------------------
        # Create CSV records
        # ----------------------------------------------------

        for bbox, label in zip(
            transformed_bboxes,
            transformed_labels,
        ):

            xmin, ymin, xmax, ymax = bbox

            xmin, ymin, xmax, ymax = clamp_bbox(
                xmin,
                ymin,
                xmax,
                ymax,
                new_width,
                new_height,
            )

            if xmax <= xmin or ymax <= ymin:
                continue

            augmented_rows.append(
                {
                    "filename": new_filename,
                    "width": new_width,
                    "height": new_height,
                    "class": label,
                    "xmin": round(xmin, 2),
                    "ymin": round(ymin, 2),
                    "xmax": round(xmax, 2),
                    "ymax": round(ymax, 2),
                    "split": "train",
                }
            )

        generated_for_class += 1

        progress.update(1)

    progress.close()

    print(
        f"Generated: {generated_for_class}"
    )

    if generated_for_class < required:

        print(
            f"WARNING: Could only generate "
            f"{generated_for_class}/{required}"
        )


# ============================================================
# CREATE FINAL DATAFRAME
# ============================================================

print()
print("=" * 70)
print("CREATING BALANCED CSV")
print("=" * 70)

augmented_df = pd.DataFrame(
    augmented_rows,
    columns=required_columns,
)

print(
    f"New augmented annotations: "
    f"{len(augmented_df):,}"
)


# Combine original CSV with augmented records.
final_df = pd.concat(
    [
        df,
        augmented_df,
    ],
    ignore_index=True,
)


# ============================================================
# SAVE CSV
# ============================================================

final_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print()
print(
    f"Balanced CSV saved to:\n"
    f"{OUTPUT_CSV}"
)


# ============================================================
# FINAL DISTRIBUTION
# ============================================================

final_train_df = final_df[
    final_df["split"].astype(str).str.lower()
    == "train"
]

after_distribution = (
    final_train_df.groupby("class")["filename"]
    .nunique()
    .rename("after_images")
    .reset_index()
)


distribution = before_distribution.merge(
    after_distribution,
    on="class",
    how="outer",
)

distribution["before_images"] = (
    distribution["before_images"]
    .fillna(0)
    .astype(int)
)

distribution["after_images"] = (
    distribution["after_images"]
    .fillna(0)
    .astype(int)
)

distribution["generated"] = (
    distribution["after_images"]
    - distribution["before_images"]
)

distribution = distribution.sort_values(
    "after_images"
)

distribution.to_csv(
    DISTRIBUTION_CSV,
    index=False,
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL TRAINING DISTRIBUTION")
print("=" * 70)

print(
    distribution.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(
    f"Original annotations: "
    f"{len(df):,}"
)

print(
    f"Augmented annotations: "
    f"{len(augmented_df):,}"
)

print(
    f"Final annotations: "
    f"{len(final_df):,}"
)

print(
    f"Original unique images: "
    f"{df['filename'].nunique():,}"
)

print(
    f"Final unique images: "
    f"{final_df['filename'].nunique():,}"
)

print()
print(
    f"Images saved in:\n"
    f"{IMAGE_OUTPUT_DIR}"
)

print()
print(
    f"Balanced CSV:\n"
    f"{OUTPUT_CSV}"
)

print()
print(
    f"Distribution report:\n"
    f"{DISTRIBUTION_CSV}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
