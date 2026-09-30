from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path("/home/admin1/Abishek-Intern/new_atr/aircrafts")

MODEL_PATH = Path(
    "/home/admin1/Abishek-Intern/new_atr/"
    "models/yolov8s_832_balanced800/weights/best.pt"
)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model:")
print(MODEL_PATH)

model = YOLO(str(MODEL_PATH))

print("\nModel loaded.")

# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_IDS = {
    "F2": 33,
    "An225": 8,
    "J35": 42,
    "YF23": 98,
    "EMB314": 27,
    "An22": 7,
    "Mi28": 61,
    "Mi26": 60,
    "Su47": 74,
    "KF21": 50,
    "KJ600": 52,
    "TB2": 78,
    "WZ7": 91,
}

# ============================================================
# CHECK MODEL CLASSES
# ============================================================

print("\nChecking model classes...")

for class_name, class_id in CLASS_IDS.items():

    if class_id not in model.names:
        print(
            f"WARNING: class ID {class_id} "
            f"({class_name}) not found in model."
        )
    else:
        model_name = model.names[class_id]

        print(
            f"{class_name:8s} -> ID {class_id:3d} "
            f"-> model name: {model_name}"
        )

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
# PROCESS EACH CLASS FOLDER
# ============================================================

for class_dir in sorted(ROOT_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    if class_name not in CLASS_IDS:
        print(
            f"\nSkipping unknown folder: {class_name}"
        )
        continue

    target_class_id = CLASS_IDS[class_name]

    print("\n" + "=" * 70)
    print(f"CLASS: {class_name}")
    print(f"CLASS ID: {target_class_id}")
    print("=" * 70)

    image_files = sorted(
        [
            p
            for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    print(f"Images: {len(image_files)}")

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for index, image_path in enumerate(image_files, start=1):

        total_images += 1

        csv_path = image_path.with_suffix(".csv")

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name}",
            end=" "
        )

        # ----------------------------------------------------
        # DO NOT OVERWRITE EXISTING CSV
        # ----------------------------------------------------

        if csv_path.exists():

            print("→ CSV already exists, skipping")

            existing_csv += 1
            continue

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(str(image_path))

        if image is None:

            print("→ FAILED TO READ IMAGE")

            failed_images += 1
            continue

        height, width = image.shape[:2]

        # ----------------------------------------------------
        # YOLO PREDICTION
        # ----------------------------------------------------

        try:

            results = model.predict(
                source=str(image_path),
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
                device=0
            )

        except Exception as e:

            print(f"→ ERROR: {e}")

            failed_images += 1
            continue

        result = results[0]

        boxes_to_save = []

        # ----------------------------------------------------
        # EXTRACT DETECTIONS
        # ----------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Keep only expected folder class
                if cls_id != target_class_id:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # ------------------------------------------------
                # CLAMP BOX TO IMAGE
                # ------------------------------------------------

                x1 = max(0, min(int(round(x1)), width - 1))
                y1 = max(0, min(int(round(y1)), height - 1))
                x2 = max(0, min(int(round(x2)), width - 1))
                y2 = max(0, min(int(round(y2)), height - 1))

                # Ignore invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes_to_save.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence
                    )
                )

        # ----------------------------------------------------
        # NO DETECTION
        # ----------------------------------------------------

        if not boxes_to_save:

            print("→ NO DETECTION")

            no_detection += 1
            continue

        # ----------------------------------------------------
        # WRITE CSV
        # ----------------------------------------------------

        filename = image_path.stem

        with open(
            csv_path,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "filename",
                "width",
                "height",
                "class",
                "xmin",
                "ymin",
                "xmax",
                "ymax"
            ])

            for (
                x1,
                y1,
                x2,
                y2,
                confidence
            ) in boxes_to_save:

                writer.writerow([
                    filename,
                    width,
                    height,
                    class_name,
                    x1,
                    y1,
                    x2,
                    y2
                ])

        annotated_images += 1
        total_boxes += len(boxes_to_save)

        print(
            f"→ SAVED "
            f"{len(boxes_to_save)} box(es)"
        )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("AUTO-ANNOTATION COMPLETE")
print("=" * 70)

print(f"Total images checked : {total_images}")
print(f"CSV files created    : {annotated_images}")
print(f"Total boxes created  : {total_boxes}")
print(f"Already had CSV      : {existing_csv}")
print(f"No detection         : {no_detection}")
print(f"Failed images        : {failed_images}")

print("=" * 70)
