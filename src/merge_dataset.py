from pathlib import Path
import pandas as pd
import shutil
import random

# ============================================================
# PATHS
# ============================================================

BASE = Path("/home/admin1/Abishek-Intern/new_atr")

ORIGINAL = BASE / "data/raw/dataset"
MASTER_CSV = BASE / "data/raw/labels_with_split.csv"

NEW_DATA = BASE / "aircrafts"

MERGED = BASE / "data/merged"
MERGED_IMAGES = MERGED / "images"
MERGED_ANNOTATIONS = MERGED / "annotations"
MERGED_MASTER = MERGED / "labels_with_split.csv"

# ============================================================
# SETTINGS
# ============================================================

SEED = 42

TRAIN_RATIO = 0.7510
VAL_RATIO = 0.1822
TEST_RATIO = 0.0668

random.seed(SEED)

# ============================================================
# CREATE DIRECTORIES
# ============================================================

MERGED_IMAGES.mkdir(parents=True, exist_ok=True)
MERGED_ANNOTATIONS.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: COPY ORIGINAL DATA
# ============================================================

print("\n========================================")
print("STEP 1: COPYING ORIGINAL DATA")
print("========================================")

original_images = 0
original_csvs = 0

for item in ORIGINAL.iterdir():

    if not item.is_file():
        continue

    suffix = item.suffix.lower()

    if suffix in [".jpg", ".jpeg", ".png", ".webp"]:

        shutil.copy2(
            item,
            MERGED_IMAGES / item.name
        )

        original_images += 1

    elif suffix == ".csv":

        # Keep original per-image CSV annotations
        shutil.copy2(
            item,
            MERGED_ANNOTATIONS / item.name
        )

        original_csvs += 1

print(f"Original images copied : {original_images}")
print(f"Original CSVs copied   : {original_csvs}")

# ============================================================
# STEP 2: LOAD ORIGINAL MASTER CSV
# ============================================================

print("\n========================================")
print("STEP 2: LOADING MASTER CSV")
print("========================================")

original_df = pd.read_csv(MASTER_CSV)

print(f"Original annotation rows : {len(original_df)}")
print(f"Original unique images   : {original_df['filename'].nunique()}")

# ============================================================
# STEP 3: COLLECT NEW IMAGES
# ============================================================

print("\n========================================")
print("STEP 3: COLLECTING NEW DATA")
print("========================================")

new_items = []

for folder in sorted(NEW_DATA.iterdir()):

    if not folder.is_dir():
        continue

    images = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ])

    print(f"{folder.name:10s}: {len(images)} images")

    for image in images:

        csv_file = image.with_suffix(".csv")

        if not csv_file.exists():
            raise FileNotFoundError(
                f"Missing CSV for {image}"
            )

        new_items.append({
            "class": folder.name,
            "image": image,
            "csv": csv_file
        })

print(f"\nTotal new images: {len(new_items)}")

# ============================================================
# STEP 4: ASSIGN TRAIN / VALIDATION / TEST
# ============================================================

print("\n========================================")
print("STEP 4: ASSIGNING SPLITS")
print("========================================")

# Stratified by aircraft class
# This ensures every new aircraft class gets approximately
# the same 75.10 / 18.22 / 6.68 distribution.

by_class = {}

for item in new_items:
    by_class.setdefault(item["class"], []).append(item)

for aircraft_class in by_class:

    items = by_class[aircraft_class]

    random.shuffle(items)

    n = len(items)

    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)

    # Make sure everything else goes to test
    n_test = n - n_train - n_val

    for i, item in enumerate(items):

        if i < n_train:
            item["split"] = "train"

        elif i < n_train + n_val:
            item["split"] = "validation"

        else:
            item["split"] = "test"

# ============================================================
# STEP 5: COPY NEW IMAGES + CSVs
# ============================================================

print("\n========================================")
print("STEP 5: COPYING NEW DATA")
print("========================================")

new_rows = []

for item in new_items:

    image = item["image"]
    csv_file = item["csv"]

    # --------------------------------------------------------
    # Prevent filename collision
    # --------------------------------------------------------

    destination_name = image.stem + image.suffix

    destination_image = MERGED_IMAGES / destination_name
    destination_csv = MERGED_ANNOTATIONS / (image.stem + ".csv")

    if destination_image.exists():

        # Add aircraft class prefix if collision occurs
        destination_name = (
            f"{item['class']}_{image.name}"
        )

        destination_image = MERGED_IMAGES / destination_name

        destination_csv = (
            MERGED_ANNOTATIONS /
            f"{item['class']}_{image.stem}.csv"
        )

    shutil.copy2(
        image,
        destination_image
    )

    # --------------------------------------------------------
    # Read annotation
    # --------------------------------------------------------

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():

        new_rows.append({
            "filename": destination_image.stem,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "class": row["class"],
            "xmin": int(row["xmin"]),
            "ymin": int(row["ymin"]),
            "xmax": int(row["xmax"]),
            "ymax": int(row["ymax"]),
            "split": item["split"]
        })

    # Copy original per-image CSV
    shutil.copy2(
        csv_file,
        destination_csv
    )

# ============================================================
# STEP 6: CREATE NEW MASTER CSV
# ============================================================

print("\n========================================")
print("STEP 6: CREATING MERGED MASTER CSV")
print("========================================")

new_df = pd.DataFrame(
    new_rows,
    columns=[
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "split"
    ]
)

merged_df = pd.concat(
    [original_df, new_df],
    ignore_index=True
)

merged_df.to_csv(
    MERGED_MASTER,
    index=False
)

# ============================================================
# STEP 7: FINAL STATISTICS
# ============================================================

print("\n========================================")
print("FINAL DATASET")
print("========================================")

print(f"Original images       : {original_images}")
print(f"New images            : {len(new_items)}")
print(f"Total merged images   : {original_images + len(new_items)}")

print()

print(f"Original annotations   : {len(original_df)}")
print(f"New annotations        : {len(new_df)}")
print(f"Total annotations      : {len(merged_df)}")

print("\n--- New Data Split ---")

print(
    new_df.groupby("split")["filename"]
    .nunique()
)

print("\n--- New Annotation Split ---")

print(
    new_df["split"].value_counts()
)

print("\n--- New Classes ---")

print(
    new_df.groupby("class")["filename"]
    .nunique()
    .sort_index()
)

print("\n--- All Classes ---")

print(
    merged_df["class"].nunique(),
    "classes"
)

print("\n========================================")
print("MERGE COMPLETE")
print("========================================")

print(f"Images      : {MERGED_IMAGES}")
print(f"Annotations : {MERGED_ANNOTATIONS}")
print(f"Master CSV  : {MERGED_MASTER}")
