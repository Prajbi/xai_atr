from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/tejas"
)

MODEL_PATH = Path(
    "/home/admin1/Abishek-Intern/new_atr/main/models/yolo_aircraft.pt"
)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

# Tejas class ID in the trained model.
# VERIFY this ID against model.names before running.
TEJAS_CLASS_ID = None

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# VALIDATE PATHS
# ============================================================

if not ROOT_DIR.is_dir():
    raise FileNotFoundError(
        f"Tejas image directory not found: {ROOT_DIR}"
    )

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model weights not found: {MODEL_PATH}"
    )

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO model:")
print(MODEL_PATH)

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully.")

# ============================================================
# FIND TEJAS CLASS ID
# ============================================================

print("\nModel class names:")

for class_id, class_name in model.names.items():
    print(f"ID {class_id:3d}: {class_name}")

# Find Tejas by its name rather than guessing its ID.
tejas_matches = [
    class_id
    for class_id, class_name in model.names.items()
    if str(class_name).strip().lower() == "tejas"
]

if TEJAS_CLASS_ID is None:

    if len(tejas_matches) == 1:
        TEJAS_CLASS_ID = tejas_matches[0]

    else:
        raise ValueError(
            "\nCould not uniquely identify the Tejas class in "
            "model.names.\n"
            "Check the class list printed above. If Tejas is "
            "present under a different name, update the matching "
            "logic to use its exact model name."
        )

print(f"\nTejas class ID: {TEJAS_CLASS_ID}")
print(f"Tejas model name: {model.names[TEJAS_CLASS_ID]}")

# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted(
    path
    for path in ROOT_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_files:
    raise RuntimeError(
        f"No images found in {ROOT_DIR}"
    )

print(f"\nTejas images found: {len(image_files)}")

# ============================================================
# STATISTICS
# ============================================================

total_images = 0
annotated_images = 0
no_detection = 0
existing_csv = 0
failed_images = 0
total_boxes = 0

# ============================================================
# PROCESS IMAGES
# ============================================================

for index, image_path in enumerate(image_files, start=1):

    total_images += 1

    csv_path = image_path.with_suffix(".csv")

    print(
        f"[{index}/{len(image_files)}] "
        f"{image_path.name}",
        end=" "
    )

    # Do not overwrite existing annotations.
    if csv_path.exists():
        print("-> CSV already exists, skipping")
        existing_csv += 1
        continue

    # Read image and determine dimensions.
    image = cv2.imread(str(image_path))

    if image is None:
        print("-> FAILED TO READ IMAGE")
        failed_images += 1
        continue

    height, width = image.shape[:2]

    # ========================================================
    # YOLO PREDICTION
    # ========================================================

    try:
        results = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            device=0
        )

    except Exception as error:
        print(f"-> PREDICTION ERROR: {error}")
        failed_images += 1
        continue

    result = results[0]
    boxes_to_save = []

    # ========================================================
    # FILTER TEJAS DETECTIONS
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Keep only detections classified as Tejas.
            if class_id != TEJAS_CLASS_ID:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Clip coordinates to the image boundaries.
            x1 = max(0, min(int(round(x1)), width - 1))
            y1 = max(0, min(int(round(y1)), height - 1))
            x2 = max(0, min(int(round(x2)), width - 1))
            y2 = max(0, min(int(round(y2)), height - 1))

            # Ignore invalid boxes.
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_to_save.append(
                (x1, y1, x2, y2, confidence)
            )

    # ========================================================
    # HANDLE NO DETECTION
    # ========================================================

    if not boxes_to_save:
        print("-> NO TEJAS DETECTION")
        no_detection += 1
        continue

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "width",
            "height",
            "class",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "confidence"
        ])

        for x1, y1, x2, y2, confidence in boxes_to_save:

            writer.writerow([
                image_path.stem,
                width,
                height,
                "Tejas",
                x1,
                y1,
                x2,
                y2,
                round(confidence, 6)
            ])

    annotated_images += 1
    total_boxes += len(boxes_to_save)

    print(
        f"-> SAVED {len(boxes_to_save)} box(es)"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("TEJAS AUTO-ANNOTATION COMPLETE")
print("=" * 65)

print(f"Total images checked : {total_images}")
print(f"CSV files created    : {annotated_images}")
print(f"Total boxes created  : {total_boxes}")
print(f"Already had CSV      : {existing_csv}")
print(f"No Tejas detection   : {no_detection}")
print(f"Failed images        : {failed_images}")

print("=" * 65)
print(f"\nAnnotations saved alongside images in:\n{ROOT_DIR}")
