import os
import shutil
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

SOURCE = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/yolo_78class_balanced_aug"
)

DEST = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/yolo_47class"
)

# ============================================================
# ORIGINAL 78-CLASS NAMES
# IMPORTANT: These must match your existing 78-class data.yaml
# ============================================================

OLD_CLASSES = [
    "A10",
    "A400M",
    "AG600",
    "AH64",
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
    "H6",
    "Il76",
    "J10",
    "J20",
    "J35",
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
    "P3",
    "RQ4",
    "Rafale",
    "SR71",
    "Su24",
    "Su25",
    "Su34",
    "Su47",
    "Su57",
    "TB2",
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
    "XB70",
    "Y20",
    "YF23",
]

# ============================================================
# YOUR 47 SELECTED CLASSES
# New IDs MUST be 0-46
# ============================================================

NEW_CLASSES = [
    "A10",
    "A400M",
    "AG600",
    "AV8B",
    "B1",
    "B2",
    "B52",
    "Be200",
    "C130",
    "C17",
    "C5",
    "E2",
    "EF2000",
    "F117",
    "F14",
    "F15",
    "F16",
    "F18",
    "F22",
    "F35",
    "F4",
    "J20",
    "JAS39",
    "MQ9",
    "Mig31",
    "Mirage2000",
    "RQ4",
    "Rafale",
    "SR71",
    "Su34",
    "Su57",
    "Tornado",
    "Tu160",
    "Tu95",
    "U2",
    "US2",
    "V22",
    "Vulcan",
    "XB70",
    "YF23",
    "Su25",
    "Su24",
    "P3",
    "C2",
    "J10",
    "E7",
    "KC135",
]

# ============================================================
# CREATE OLD -> NEW ID MAPPING
# ============================================================

old_id = {name: i for i, name in enumerate(OLD_CLASSES)}
new_id = {name: i for i, name in enumerate(NEW_CLASSES)}

ID_MAPPING = {
    old_id[name]: new_id[name]
    for name in NEW_CLASSES
}

print("=" * 70)
print("47-CLASS DATASET CREATION")
print("=" * 70)

print(f"\nSource : {SOURCE}")
print(f"Output : {DEST}")

print("\nClass mapping:")
for name in NEW_CLASSES:
    print(
        f"{name:12s}: "
        f"old={old_id[name]:2d} -> "
        f"new={new_id[name]:2d}"
    )

# ============================================================
# CREATE DESTINATION DIRECTORIES
# ============================================================

for split in ["train", "validation", "test"]:
    (DEST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DEST / "labels" / split).mkdir(parents=True, exist_ok=True)

# ============================================================
# PROCESS DATASET
# ============================================================

statistics = {
    split: {
        "images_seen": 0,
        "images_copied": 0,
        "images_skipped": 0,
        "labels_written": 0,
        "objects_kept": 0,
        "objects_removed": 0,
    }
    for split in ["train", "validation", "test"]
}

class_counts = {
    split: {name: 0 for name in NEW_CLASSES}
    for split in ["train", "validation", "test"]
}

for split in ["train", "validation", "test"]:

    source_images = SOURCE / "images" / split
    source_labels = SOURCE / "labels" / split

    if not source_images.exists():
        print(f"\nWARNING: Missing image directory: {source_images}")
        continue

    if not source_labels.exists():
        print(f"\nWARNING: Missing label directory: {source_labels}")
        continue

    image_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(source_images.glob(ext))

    print(f"\n{'-' * 70}")
    print(f"Processing {split}")
    print(f"Images found: {len(image_files)}")
    print(f"{'-' * 70}")

    for image_path in image_files:

        statistics[split]["images_seen"] += 1

        label_path = source_labels / f"{image_path.stem}.txt"

        # ----------------------------------------------------
        # No label file
        # ----------------------------------------------------

        if not label_path.exists():
            statistics[split]["images_skipped"] += 1
            continue

        new_annotations = []

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"WARNING: Invalid label format: "
                    f"{label_path}"
                )
                continue

            try:
                old_class = int(parts[0])
            except ValueError:
                continue

            # ------------------------------------------------
            # Keep only selected classes
            # ------------------------------------------------

            if old_class not in ID_MAPPING:
                statistics[split]["objects_removed"] += 1
                continue

            new_class = ID_MAPPING[old_class]

            # Keep YOLO coordinates unchanged
            x_center = parts[1]
            y_center = parts[2]
            width = parts[3]
            height = parts[4]

            new_line = (
                f"{new_class} "
                f"{x_center} "
                f"{y_center} "
                f"{width} "
                f"{height}\n"
            )

            new_annotations.append(new_line)

            class_name = NEW_CLASSES[new_class]
            class_counts[split][class_name] += 1
            statistics[split]["objects_kept"] += 1

        # ----------------------------------------------------
        # Skip image if it has no selected aircraft
        # ----------------------------------------------------

        if not new_annotations:
            statistics[split]["images_skipped"] += 1
            continue

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = DEST / "images" / split / image_path.name

        shutil.copy2(
            image_path,
            destination_image
        )

        # ----------------------------------------------------
        # Write remapped label
        # ----------------------------------------------------

        destination_label = (
            DEST / "labels" / split / f"{image_path.stem}.txt"
        )

        with open(destination_label, "w") as f:
            f.writelines(new_annotations)

        statistics[split]["images_copied"] += 1
        statistics[split]["labels_written"] += 1

# ============================================================
# CREATE DATA.YAML
# ============================================================

yaml_path = DEST / "data.yaml"

with open(yaml_path, "w") as f:

    f.write(f"path: {DEST}\n")
    f.write("train: images/train\n")
    f.write("val: images/validation\n")
    f.write("test: images/test\n\n")
    f.write(f"nc: {len(NEW_CLASSES)}\n")
    f.write("names:\n")

    for name in NEW_CLASSES:
        f.write(f"  - {name}\n")

# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("DATASET CREATION COMPLETE")
print("=" * 70)

print(f"\nOutput dataset:")
print(DEST)

print("\nSplit statistics:")

for split in ["train", "validation", "test"]:

    s = statistics[split]

    print(f"\n{split.upper()}")
    print(f"  Images seen    : {s['images_seen']}")
    print(f"  Images copied  : {s['images_copied']}")
    print(f"  Images skipped : {s['images_skipped']}")
    print(f"  Labels written : {s['labels_written']}")
    print(f"  Objects kept   : {s['objects_kept']}")
    print(f"  Objects removed: {s['objects_removed']}")

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    f"\n{'Class':<15}"
    f"{'Train':>10}"
    f"{'Validation':>12}"
    f"{'Test':>10}"
    f"{'Total':>10}"
)

print("-" * 60)

for name in NEW_CLASSES:

    train_count = class_counts["train"][name]
    val_count = class_counts["validation"][name]
    test_count = class_counts["test"][name]

    total = train_count + val_count + test_count

    print(
        f"{name:<15}"
        f"{train_count:>10}"
        f"{val_count:>12}"
        f"{test_count:>10}"
        f"{total:>10}"
    )

print("\n")
print("=" * 70)
print("47 classes:", len(NEW_CLASSES))
print("YAML:", yaml_path)
print("=" * 70)
