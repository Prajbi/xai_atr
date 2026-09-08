from pathlib import Path
import pandas as pd

BASE_DIR = Path("/home/admin1/Abishek-Intern/new_atr/data/raw")

CSV_PATH = BASE_DIR / "labels_with_split.csv"
IMAGE_DIR = BASE_DIR / "dataset"

TARGET = 800

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
}

# ------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

df["filename"] = df["filename"].astype(str)
df["class"] = df["class"].astype(str)
df["split"] = df["split"].astype(str).str.lower()

train_df = df[df["split"] == "train"].copy()

# ------------------------------------------------------------
# Build image index
# ONLY actual image files
# ------------------------------------------------------------

image_index = {}

for p in IMAGE_DIR.iterdir():

    if not p.is_file():
        continue

    if p.suffix not in VALID_IMAGE_EXTENSIONS:
        continue

    image_index[p.stem] = p

# ------------------------------------------------------------
# Check training images
# ------------------------------------------------------------

train_images = set(
    train_df["filename"].unique()
)

matched = train_images & set(image_index.keys())
missing = train_images - set(image_index.keys())

print("=" * 70)
print("TRAINING IMAGE VERIFICATION")
print("=" * 70)

print(f"Training unique images : {len(train_images):,}")
print(f"Actual image files     : {len(image_index):,}")
print(f"Matched training       : {len(matched):,}")
print(f"Missing training       : {len(missing):,}")

if missing:
    print("\nFirst missing training images:")

    for x in sorted(missing)[:20]:
        print(" ", x)

# ------------------------------------------------------------
# Class distribution
# ------------------------------------------------------------

class_counts = (
    train_df.groupby("class")["filename"]
    .nunique()
    .sort_values()
)

minority = class_counts[
    class_counts < TARGET
]

print()
print("=" * 70)
print("MINORITY CLASSES")
print("=" * 70)

print(f"Total classes        : {len(class_counts)}")
print(f"Classes below {TARGET} : {len(minority)}")

print()

print(
    f"{'CLASS':25s}"
    f"{'CURRENT':>10s}"
    f"{'GENERATE':>12s}"
    f"{'TARGET':>10s}"
)

print("-" * 60)

total_to_generate = 0

for cls, count in minority.items():

    generate = TARGET - count

    total_to_generate += generate

    print(
        f"{cls:25s}"
        f"{count:10d}"
        f"{generate:12d}"
        f"{TARGET:10d}"
    )

print("-" * 60)

print(
    f"{'TOTAL':25s}"
    f"{'':10s}"
    f"{total_to_generate:12d}"
)

# ------------------------------------------------------------
# Annotation statistics
# ------------------------------------------------------------

print()
print("=" * 70)
print("ANNOTATION STATISTICS")
print("=" * 70)

print(
    f"Training annotations : {len(train_df):,}"
)

print(
    f"Training images      : "
    f"{train_df['filename'].nunique():,}"
)

print(
    f"Training classes     : "
    f"{train_df['class'].nunique():,}"
)

# ------------------------------------------------------------
# File-type check
# ------------------------------------------------------------

print()
print("=" * 70)
print("FILES IN raw/dataset")
print("=" * 70)

extension_counts = {}

for p in IMAGE_DIR.iterdir():

    if p.is_file():

        ext = p.suffix.lower()

        extension_counts[ext] = (
            extension_counts.get(ext, 0) + 1
        )

for ext, count in sorted(
    extension_counts.items(),
    key=lambda x: -x[1]
):

    print(
        f"{ext or '[no extension]':10s} "
        f"{count:,}"
    )

print()
print("=" * 70)
print("DRY RUN COMPLETE")
print("=" * 70)
