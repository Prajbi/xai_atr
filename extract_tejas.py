import csv
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path("/home/admin1/Abishek-Intern/new_atr")

SOURCE_ROOT = PROJECT / "data/raw/yolo"
MASTER_CSV = PROJECT / "data/raw/labels_with_split.csv"

OUTPUT_ROOT = PROJECT / "data/tejas_only"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

TARGET_CLASS_NAMES = {"tejas", "79"}

SPLITS = ("train", "validation", "test")

# ============================================================
# VALIDATE INPUTS
# ============================================================

if not MASTER_CSV.exists():
    raise FileNotFoundError(
        f"Master CSV not found: {MASTER_CSV}"
    )

SOURCE_IMAGES = SOURCE_ROOT / "images"

if not SOURCE_IMAGES.exists():
    raise FileNotFoundError(
        f"Source images directory not found: {SOURCE_IMAGES}"
    )

print("=" * 65)
print("TEJAS DATASET EXTRACTION")
print("=" * 65)

print(f"Source images: {SOURCE_IMAGES}")
print(f"Master CSV:    {MASTER_CSV}")
print(f"Output:        {OUTPUT_ROOT}")

# ============================================================
# INDEX SOURCE IMAGES
# ============================================================

print("\nIndexing source images...")

image_index = defaultdict(list)

for image_path in SOURCE_IMAGES.rglob("*"):

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    # CSV identifiers may omit the image extension.
    image_index[image_path.stem].append(image_path)

indexed_count = sum(len(paths) for paths in image_index.values())

print(f"Total indexed image files: {indexed_count}")
print(f"Unique image identifiers:  {len(image_index)}")

# ============================================================
# READ MASTER CSV
# ============================================================

print("\nReading master CSV...")

with open(
    MASTER_CSV,
    "r",
    newline="",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    required_columns = {
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "split"
    }

    actual_columns = set(reader.fieldnames or [])

    if not required_columns.issubset(actual_columns):
        raise ValueError(
            "Master CSV is missing required columns.\n"
            f"Required: {sorted(required_columns)}\n"
            f"Found: {reader.fieldnames}"
        )

    rows = list(reader)

print(f"Master CSV annotation rows: {len(rows)}")

# ============================================================
# EXTRACT TEJAS ANNOTATIONS
# ============================================================

tejas_annotations = defaultdict(list)
image_metadata = {}

invalid_rows = 0

for row in rows:

    class_name = str(row["class"]).strip().lower()

    if class_name not in TARGET_CLASS_NAMES:
        continue

    # Remove directory components and optional extension.
    filename = Path(
        str(row["filename"]).strip()
    ).stem

    try:
        width = int(float(row["width"]))
        height = int(float(row["height"]))

        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])

    except (ValueError, TypeError):
        invalid_rows += 1
        continue

    if width <= 0 or height <= 0:
        invalid_rows += 1
        continue

    # Clip bounding boxes to image boundaries.
    xmin = max(0.0, min(xmin, width))
    ymin = max(0.0, min(ymin, height))
    xmax = max(0.0, min(xmax, width))
    ymax = max(0.0, min(ymax, height))

    if xmax <= xmin or ymax <= ymin:
        invalid_rows += 1
        continue

    original_split = str(row["split"]).strip().lower()

    # Keep the requested output directory name.
    if original_split in {"val", "valid", "validation"}:
        split = "validation"

    elif original_split == "train":
        split = "train"

    elif original_split == "test":
        split = "test"

    else:
        print(
            f"WARNING: Unknown split '{original_split}' "
            f"for {filename}"
        )
        invalid_rows += 1
        continue

    tejas_annotations[filename].append(
        (xmin, ymin, xmax, ymax)
    )

    # Ensure all annotations for an image agree on metadata.
    metadata = {
        "width": width,
        "height": height,
        "split": split
    }

    if filename in image_metadata:
        existing = image_metadata[filename]

        if existing != metadata:
            raise ValueError(
                f"Inconsistent metadata for {filename}:\n"
                f"{existing}\n{metadata}"
            )
    else:
        image_metadata[filename] = metadata

print(f"\nUnique Tejas image IDs: {len(tejas_annotations)}")

total_tejas_boxes = sum(
    len(boxes) for boxes in tejas_annotations.values()
)

print(f"Tejas bounding boxes:   {total_tejas_boxes}")
print(f"Invalid rows skipped:   {invalid_rows}")

# ============================================================
# VALIDATE IMAGE MATCHES BEFORE EXPORTING
# ============================================================

print("\nChecking image matches...")

missing_images = []
ambiguous_images = []

for filename in tejas_annotations:

    matches = image_index.get(filename, [])

    if not matches:
        missing_images.append(filename)

    elif len(matches) > 1:
        # Multiple files with the same stem can lead to
        # incorrect matches, so stop rather than guess.
        ambiguous_images.append((filename, matches))

print(f"Matching image IDs: {len(tejas_annotations) - len(missing_images)}")
print(f"Missing image IDs:  {len(missing_images)}")
print(f"Ambiguous IDs:      {len(ambiguous_images)}")

if missing_images:

    print("\nExamples of missing image IDs:")

    for filename in missing_images[:15]:
        print(" ", filename)

if ambiguous_images:

    print("\nExamples of ambiguous image IDs:")

    for filename, paths in ambiguous_images[:10]:
        print(f"  {filename}")

        for path in paths:
            print(f"    {path}")

# Stop if image matching is incomplete or ambiguous.
# This prevents exporting a misleadingly incomplete dataset.
if missing_images or ambiguous_images:

    raise RuntimeError(
        "\nImage matching is incomplete or ambiguous. "
        "No extraction was performed.\n"
        "Check the missing/ambiguous identifiers above before "
        "continuing. The source dataset has not been modified."
    )

# ============================================================
# PREPARE OUTPUT DIRECTORIES
# ============================================================

print("\nPreparing output directories...")

# Create directories without deleting previous files.
for split in SPLITS:

    (OUTPUT_ROOT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

(OUTPUT_ROOT / "annotations").mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# EXPORT IMAGES, YOLO LABELS AND CSV
# ============================================================

print("\nExporting Tejas images and labels...")

exported_images = 0
exported_boxes = 0

split_image_counts = defaultdict(int)
split_box_counts = defaultdict(int)

csv_rows = []

for filename, boxes in tejas_annotations.items():

    source_image = image_index[filename][0]
    metadata = image_metadata[filename]

    split = metadata["split"]
    width = metadata["width"]
    height = metadata["height"]

    # Preserve original image extension.
    destination_image = (
        OUTPUT_ROOT / "images" / split / source_image.name
    )

    destination_label = (
        OUTPUT_ROOT / "labels" / split /
        f"{source_image.stem}.txt"
    )

    # Verify source dimensions agree with CSV dimensions.
    # Requires OpenCV in the current Python environment.
    try:
        import cv2

        image = cv2.imread(str(source_image))

        if image is None:
            raise RuntimeError(
                f"Cannot read source image: {source_image}"
            )

        actual_height, actual_width = image.shape[:2]

        if actual_width != width or actual_height != height:
            raise ValueError(
                f"Dimension mismatch for {filename}: "
                f"CSV={width}x{height}, "
                f"actual={actual_width}x{actual_height}"
            )

    except ImportError:
        print(
            "WARNING: OpenCV is unavailable; "
            "image dimensions could not be verified."
        )

    shutil.copy2(source_image, destination_image)

    with open(
        destination_label,
        "w",
        encoding="utf-8"
    ) as label_file:

        for xmin, ymin, xmax, ymax in boxes:

            # Normalize coordinates for YOLO format.
            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            # Single-class dataset: Tejas = class ID 0.
            label_file.write(
                f"0 {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}\n"
            )

            # CSV stores pixel coordinates and original filename ID.
            csv_rows.append({
                "filename": filename,
                "width": width,
                "height": height,
                "class": "Tejas",
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "split": split
            })

            exported_boxes += 1
            split_box_counts[split] += 1

    exported_images += 1
    split_image_counts[split] += 1

# ============================================================
# SAVE TEJAS MASTER CSV
# ============================================================

csv_path = (
    OUTPUT_ROOT / "annotations" / "tejas_only.csv"
)

csv_columns = [
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

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_columns
    )

    writer.writeheader()
    writer.writerows(csv_rows)

# ============================================================
# CREATE YOLO DATA YAML
# ============================================================

yaml_path = OUTPUT_ROOT / "data.yaml"

yaml_content = f"""path: {OUTPUT_ROOT}
train: images/train
val: images/validation
test: images/test
nc: 1
names:
  0: Tejas
"""

yaml_path.write_text(
    yaml_content,
    encoding="utf-8"
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("TEJAS EXTRACTION COMPLETE")
print("=" * 65)

print(f"Output directory: {OUTPUT_ROOT}")
print(f"Images exported:  {exported_images}")
print(f"Boxes exported:   {exported_boxes}")

print(f"\nMaster CSV: {csv_path}")
print(f"YOLO YAML:  {yaml_path}")

print("\nSplit-wise statistics:")

for split in SPLITS:

    print(
        f"{split:12} "
        f"Images: {split_image_counts[split]:4} | "
        f"Boxes: {split_box_counts[split]:4}"
    )

print("\nExpected output structure:")

for path in [
    OUTPUT_ROOT / "images" / "train",
    OUTPUT_ROOT / "images" / "validation",
    OUTPUT_ROOT / "images" / "test",
    OUTPUT_ROOT / "labels" / "train",
    OUTPUT_ROOT / "labels" / "validation",
    OUTPUT_ROOT / "labels" / "test"
]:
    print(path)

print("\nOriginal dataset was not modified.")
