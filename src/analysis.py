import pandas as pd
from pathlib import Path
CSV_PATH = Path("data/raw/labels_with_split.csv")
df = pd.read_csv(CSV_PATH)
print("=" * 70)
print("MILITARY AIRCRAFT DATASET ANALYSIS")
print("=" * 70)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# NUMBER OF CLASSES
# ============================================================

classes = sorted(
    df["class"].astype(str).unique()
)

print("\nNumber of aircraft classes:")
print(len(classes))


print("\nAircraft classes:")

for i, cls in enumerate(classes):
    print(f"{i:3d} : {cls}")


# ============================================================
# INSTANCE DISTRIBUTION
# ============================================================

instance_counts = (
    df["class"]
    .value_counts()
    .sort_values(ascending=False)
)


print("\n" + "=" * 70)
print("AIRCRAFT INSTANCE DISTRIBUTION")
print("=" * 70)

print(instance_counts.to_string())


# ============================================================
# UNIQUE IMAGE DISTRIBUTION
# ============================================================

image_counts = (
    df.groupby("class")["filename"]
    .nunique()
    .sort_values(ascending=False)
)


print("\n" + "=" * 70)
print("UNIQUE IMAGE DISTRIBUTION")
print("=" * 70)

print(image_counts.to_string())


# ============================================================
# INSTANCE IMBALANCE
# ============================================================

max_instances = instance_counts.max()
min_instances = instance_counts.min()

instance_ratio = (
    max_instances / min_instances
)


# ============================================================
# IMAGE IMBALANCE
# ============================================================

max_images = image_counts.max()
min_images = image_counts.min()

image_ratio = (
    max_images / min_images
)


print("\n" + "=" * 70)
print("IMBALANCE")
print("=" * 70)

print(
    f"Maximum aircraft instances : "
    f"{max_instances}"
)

print(
    f"Minimum aircraft instances : "
    f"{min_instances}"
)

print(
    f"Instance imbalance ratio   : "
    f"{instance_ratio:.2f}"
)

print(
    f"\nMaximum unique images      : "
    f"{max_images}"
)

print(
    f"Minimum unique images      : "
    f"{min_images}"
)

print(
    f"Image imbalance ratio      : "
    f"{image_ratio:.2f}"
)


# ============================================================
# TRAIN / VALIDATION / TEST DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION BY SPLIT")
print("=" * 70)

split_class_counts = pd.crosstab(
    df["class"],
    df["split"]
)

print(
    split_class_counts.to_string()
)


# ============================================================
# SPLIT SIZES
# ============================================================

print("\n" + "=" * 70)
print("SPLIT SIZES")
print("=" * 70)

split_counts = df["split"].value_counts()

for split, count in split_counts.items():

    percentage = (
        count / len(df) * 100
    )

    print(
        f"{split:12s}: "
        f"{count:6d} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# CLASS COUNTS PER SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CLASS STATISTICS PER SPLIT")
print("=" * 70)

for split in ["train", "validation", "test"]:

    split_df = df[
        df["split"] == split
    ]

    counts = (
        split_df["class"]
        .value_counts()
    )

    print(f"\n{split.upper()}")

    print(
        f"Samples       : {len(split_df)}"
    )

    print(
        f"Classes       : {split_df['class'].nunique()}"
    )

    print(
        f"Maximum class : {counts.max()}"
    )

    print(
        f"Minimum class : {counts.min()}"
    )

    print(
        f"Median        : {counts.median():.1f}"
    )

    print(
        f"Mean          : {counts.mean():.1f}"
    )


# ============================================================
# RAREST CLASSES
# ============================================================

print("\n" + "=" * 70)
print("10 RAREST CLASSES")
print("=" * 70)

print(
    instance_counts
    .sort_values()
    .head(10)
    .to_string()
)


# ============================================================
# MOST COMMON CLASSES
# ============================================================

print("\n" + "=" * 70)
print("10 MOST COMMON CLASSES")
print("=" * 70)

print(
    instance_counts
    .head(10)
    .to_string()
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Total annotations : {len(df)}"
)

print(
    f"Unique images     : {df['filename'].nunique()}"
)

print(
    f"Aircraft classes  : {len(classes)}"
)

print(
    f"Instance ratio    : {instance_ratio:.2f}x"
)

print(
    f"Image ratio       : {image_ratio:.2f}x"
)

print("=" * 70)
