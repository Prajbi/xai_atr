from pathlib import Path
import pandas as pd
import shutil
import random

# ============================================================
# PATHS
# ============================================================

SOURCE = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/raw/dataset"
)

OUTPUT = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/processed"
)

# ============================================================
# SPLIT
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

random.seed(SEED)

# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:

    (OUTPUT / split).mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# FIND CSV FILES
# ============================================================

csv_files = list(SOURCE.glob("*.csv"))

print("=" * 70)
print("DATASET SPLITTING")
print("=" * 70)

print("Source:", SOURCE)
print("Images/annotations:", len(csv_files))

# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(csv_files)

total = len(csv_files)

train_end = int(total * TRAIN_RATIO)

val_end = train_end + int(total * VAL_RATIO)

train_files = csv_files[:train_end]

val_files = csv_files[train_end:val_end]

test_files = csv_files[val_end:]

print("\nSplit:")
print("Train:", len(train_files))
print("Val  :", len(val_files))
print("Test :", len(test_files))

# ============================================================
# FIND IMAGE
# ============================================================

def find_image(stem):

    for ext in [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG"
    ]:

        image = SOURCE / f"{stem}{ext}"

        if image.exists():
            return image

    return None


# ============================================================
# COPY DATA
# ============================================================

def copy_split(files, split):

    output_dir = OUTPUT / split

    print(
        f"\nCopying {split}..."
    )

    for csv_file in files:

        # Copy CSV
        shutil.copy2(
            csv_file,
            output_dir / csv_file.name
        )

        # Find corresponding image
        image_file = find_image(
            csv_file.stem
        )

        if image_file is None:

            print(
                f"WARNING: image missing "
                f"for {csv_file.name}"
            )

            continue

        shutil.copy2(
            image_file,
            output_dir / image_file.name
        )


copy_split(train_files, "train")
copy_split(val_files, "val")
copy_split(test_files, "test")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def class_distribution(files):

    counts = {}

    for csv_file in files:

        try:
            df = pd.read_csv(csv_file)

        except Exception:
            continue

        for cls in df["class"]:

            cls = str(cls)

            counts[cls] = (
                counts.get(cls, 0) + 1
            )

    return counts


train_counts = class_distribution(train_files)
val_counts = class_distribution(val_files)
test_counts = class_distribution(test_files)


# ============================================================
# SAVE DISTRIBUTION
# ============================================================

all_classes = sorted(
    set(train_counts)
    | set(val_counts)
    | set(test_counts)
)

distribution = []

for cls in all_classes:

    distribution.append(
        {
            "class": cls,
            "train": train_counts.get(cls, 0),
            "val": val_counts.get(cls, 0),
            "test": test_counts.get(cls, 0),
            "total": (
                train_counts.get(cls, 0)
                + val_counts.get(cls, 0)
                + test_counts.get(cls, 0)
            )
        }
    )

distribution_df = pd.DataFrame(
    distribution
)

distribution_df = distribution_df.sort_values(
    "total"
)

distribution_df.to_csv(
    OUTPUT / "class_distribution.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    distribution_df.to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nOutput:"
)

print(OUTPUT)

print(
    "\nClass distribution saved to:"
)

print(
    OUTPUT / "class_distribution.csv"
)
