from pathlib import Path
from collections import Counter
import shutil
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")

CSV_PATH = (
    PROJECT_ROOT
    / "data/raw/labels_with_split.csv"
)

# Original raw images
RAW_IMAGE_DIR = (
    PROJECT_ROOT
    / "data/raw/dataset"
)

# New 89-class YOLO dataset
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data/yolo_aircraft_89"
)


# ============================================================
# EXACT 89 CLASSES REQUIRED BY YOUR MODEL
# ============================================================

CLASS_NAMES = [
    "A10",
    "A400M",
    "AG600",
    "AH64",
    "AKINCI",
    "AV8B",
    "An124",
    "An22",
    "An225",
    "An72",
    "B1",
    "B2",
    "B52",
    "Be200",
    "C1",
    "C130",
    "C17",
    "C2",
    "C390",
    "C5",
    "CH47",
    "CH53",
    "CL415",
    "E2",
    "E7",
    "EF2000",
    "EMB314",
    "F117",
    "F14",
    "F15",
    "F16",
    "F18",
    "F2",
    "F22",
    "F35",
    "F4",
    "FCK1",
    "H6",
    "Il76",
    "J10",
    "J20",
    "J35",
    "J36",
    "JAS39",
    "JF17",
    "JH7",
    "KC135",
    "KF21",
    "KJ600",
    "Ka27",
    "Ka52",
    "MQ9",
    "Mi24",
    "Mi26",
    "Mi28",
    "Mi8",
    "Mig29",
    "Mig31",
    "Mirage2000",
    "NH90",
    "P3",
    "RQ4",
    "Rafale",
    "SR71",
    "Su24",
    "Su25",
    "Su34",
    "Su47",
    "Su57",
    "T50",
    "TB001",
    "TB2",
    "Tejas",
    "Tornado",
    "Tu160",
    "Tu22M",
    "Tu95",
    "U2",
    "UH60",
    "US2",
    "V22",
    "Vulcan",
    "WZ7",
    "X32",
    "XB70",
    "Y20",
    "YF23",
    "Z10",
    "Z19",
]


# ============================================================
# CHECK
# ============================================================

assert len(CLASS_NAMES) == 89

CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}


# ============================================================
# PRINT HEADER
# ============================================================

print("\n========================================")
print("     CREATE 89-CLASS YOLO DATASET")
print("========================================")

print(f"\nCSV:")
print(CSV_PATH)

print("\nRaw images:")
print(RAW_IMAGE_DIR)

print("\nOutput:")
print(OUTPUT_ROOT)

print(f"\nTarget classes: {len(CLASS_NAMES)}")


# ============================================================
# CHECK INPUTS
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV not found:\n{CSV_PATH}"
    )

if not RAW_IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Raw image directory not found:\n{RAW_IMAGE_DIR}"
    )


# ============================================================
# LOAD CSV
# ============================================================

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
    raise RuntimeError(
        f"CSV is missing columns: {missing}"
    )


print("\n========================================")
print("          CSV INFORMATION")
print("========================================")

print(f"Rows       : {len(df):,}")
print(f"Images     : {df['filename'].nunique():,}")
print(f"Classes    : {df['class'].nunique()}")
print(f"Splits     : {df['split'].unique().tolist()}")


# ============================================================
# SHOW ORIGINAL CLASSES
# ============================================================

print("\n========================================")
print("       ORIGINAL CSV CLASSES")
print("========================================")

csv_classes = sorted(
    df["class"].dropna().unique()
)

for cls in csv_classes:
    print(cls)


# ============================================================
# CHECK UNKNOWN CLASSES
# ============================================================

unknown_classes = sorted(
    set(csv_classes) - set(CLASS_NAMES)
)

print("\n========================================")
print("        CLASSES TO BE REMOVED")
print("========================================")

if unknown_classes:

    print(
        f"\nFound {len(unknown_classes)} "
        "classes not required by the 89-class model:"
    )

    for cls in unknown_classes:
        count = (df["class"] == cls).sum()
        print(f"{cls:15s} : {count:,} annotations")

else:

    print("\nNo extra classes found.")


# ============================================================
# FILTER TO EXACT 89 CLASSES
# ============================================================

df_filtered = df[
    df["class"].isin(CLASS_TO_ID)
].copy()

print("\n========================================")
print("         FILTERED DATASET")
print("========================================")

print(
    f"Original annotations : {len(df):,}"
)

print(
    f"Kept annotations     : {len(df_filtered):,}"
)

print(
    f"Removed annotations  : "
    f"{len(df) - len(df_filtered):,}"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:

    (
        OUTPUT_ROOT
        / "images"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT_ROOT
        / "labels"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".webp",
    ".WEBP",
]


def find_image(filename):
    """
    Find the actual image corresponding to
    the filename stored in the CSV.
    """

    # Already has extension
    if Path(filename).suffix:
        direct = RAW_IMAGE_DIR / filename

        if direct.exists():
            return direct

    # Try all common extensions
    for ext in IMAGE_EXTENSIONS:

        candidate = RAW_IMAGE_DIR / f"{filename}{ext}"

        if candidate.exists():
            return candidate

    return None


# ============================================================
# STATISTICS
# ============================================================

image_stats = Counter()
class_stats = Counter()
missing_images = []


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

for split in ["train", "val", "test"]:

    print("\n========================================")
    print(f"PROCESSING {split.upper()}")
    print("========================================")

    split_df = df_filtered[
        df_filtered["split"].astype(str).str.lower() == split
    ]

    print(
        f"Annotations: {len(split_df):,}"
    )

    unique_files = split_df["filename"].unique()

    print(
        f"Images: {len(unique_files):,}"
    )

    output_image_dir = (
        OUTPUT_ROOT
        / "images"
        / split
    )

    output_label_dir = (
        OUTPUT_ROOT
        / "labels"
        / split
    )

    processed = 0

    # --------------------------------------------------------
    # GROUP BY IMAGE
    # --------------------------------------------------------

    for filename, group in split_df.groupby(
        "filename"
    ):

        image_path = find_image(str(filename))

        if image_path is None:

            missing_images.append(
                (split, filename)
            )

            continue

        # ----------------------------------------------------
        # COPY IMAGE
        # ----------------------------------------------------

        destination_image = (
            output_image_dir
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        # ----------------------------------------------------
        # CREATE YOLO LABELS
        # ----------------------------------------------------

        yolo_lines = []

        for _, row in group.iterrows():

            class_name = str(row["class"])

            class_id = CLASS_TO_ID[class_name]

            width = float(row["width"])
            height = float(row["height"])

            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            # -----------------------------------------------
            # Convert XYXY -> YOLO XYWH normalized
            # -----------------------------------------------

            x_center = (
                (xmin + xmax) / 2.0
            )

            y_center = (
                (ymin + ymax) / 2.0
            )

            box_width = xmax - xmin
            box_height = ymax - ymin

            # Normalize
            x_center /= width
            y_center /= height

            box_width /= width
            box_height /= height

            # -----------------------------------------------
            # Safety checks
            # -----------------------------------------------

            x_center = max(
                0.0,
                min(1.0, x_center)
            )

            y_center = max(
                0.0,
                min(1.0, y_center)
            )

            box_width = max(
                0.0,
                min(1.0, box_width)
            )

            box_height = max(
                0.0,
                min(1.0, box_height)
            )

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            class_stats[class_id] += 1

        # ----------------------------------------------------
        # WRITE LABEL FILE
        # ----------------------------------------------------

        label_path = (
            output_label_dir
            / f"{image_path.stem}.txt"
        )

        with open(label_path, "w") as f:

            f.write(
                "\n".join(yolo_lines)
            )

            f.write("\n")

        processed += 1

    image_stats[split] = processed

    print(
        f"Successfully processed: {processed:,}"
    )


# ============================================================
# CREATE DATA.YAML
# ============================================================

yaml_path = OUTPUT_ROOT / "data.yaml"

with open(yaml_path, "w") as f:

    f.write(
        f"path: {OUTPUT_ROOT}\n"
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

    f.write(
        "\nnc: 89\n"
    )

    f.write(
        "names:\n"
    )

    for idx, name in enumerate(CLASS_NAMES):

        f.write(
            f"  {idx}: {name}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("          DATASET COMPLETE")
print("========================================")

print(
    f"\nOutput directory:\n{OUTPUT_ROOT}"
)

print("\nImages:")

for split in ["train", "val", "test"]:

    print(
        f"  {split:5s}: "
        f"{image_stats[split]:,}"
    )


print("\nClass distribution:")

for idx, name in enumerate(CLASS_NAMES):

    print(
        f"{idx:3d} | "
        f"{name:12s} | "
        f"{class_stats[idx]:8,}"
    )


if missing_images:

    print("\n========================================")
    print("          MISSING IMAGES")
    print("========================================")

    print(
        f"Missing images: "
        f"{len(missing_images):,}"
    )

    for split, filename in missing_images[:20]:

        print(
            f"{split:5s} : {filename}"
        )

    if len(missing_images) > 20:

        print(
            f"... and "
            f"{len(missing_images) - 20} more"
        )


print("\n========================================")
print("             DATA.YAML")
print("========================================")

print(yaml_path)

print("\n========================================")
print("                 DONE")
print("========================================")
