from pathlib import Path
import shutil
from collections import Counter

# ============================================================
# PATHS
# ============================================================

BASE = Path("/home/admin1/Abishek-Intern/new_atr")

SOURCE = BASE / "data/yolo_merged"
OUTPUT = BASE / "data/yolo_78class"

# ============================================================
# REQUESTED 78 CLASSES
# ============================================================

classes = [
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
    "YF23"
]

assert len(classes) == 78
assert len(set(classes)) == 78

# ============================================================
# ORIGINAL 102-CLASS MAPPING
# ============================================================

original_classes = [
    "A10", "A400M", "AG600", "AH64", "AKINCI", "AV8B",
    "An124", "An22", "An225", "An72", "B1", "B2", "B21",
    "B52", "Be200", "C1", "C130", "C17", "C2", "C390",
    "C5", "CH47", "CH53", "CL415", "E2", "E7", "EF2000",
    "EMB314", "F117", "F14", "F15", "F16", "F18", "F2",
    "F22", "F35", "F4", "FCK1", "H6", "Il76", "J10", "J20",
    "J35", "J36", "J50", "JAS39", "JF17", "JH7", "KAAN",
    "KC135", "KF21", "KIZILELMA", "KJ600", "Ka27", "Ka52",
    "MQ20", "MQ25", "MQ28", "MQ9", "Mi24", "Mi26", "Mi28",
    "Mi8", "Mig29", "Mig31", "Mirage2000", "NH90", "P3",
    "RQ4", "Rafale", "SR71", "Su24", "Su25", "Su34", "Su47",
    "Su57", "T50", "TB001", "TB2", "Tejas", "Tornado",
    "Tu160", "Tu22M", "Tu95", "U2", "UH60", "US2", "V22",
    "V280", "Vulcan", "WZ10", "WZ7", "WZ9", "X29", "X32",
    "XB70", "XQ58", "Y20", "YF23", "Z10", "Z19", "Z21"
]

assert len(original_classes) == 102

old_id = {
    name: i
    for i, name in enumerate(original_classes)
}

new_id = {
    name: i
    for i, name in enumerate(classes)
}

# Map old YOLO ID -> new YOLO ID
id_mapping = {
    old_id[name]: new_id[name]
    for name in classes
}

print("========================================")
print("78-CLASS DATASET CREATION")
print("========================================")

print("Source :", SOURCE)
print("Output :", OUTPUT)
print("Classes:", len(classes))

print("\nID REMAPPING:")
for name in classes:
    print(
        f"{old_id[name]:3d} -> {new_id[name]:2d}  {name}"
    )

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "validation", "test"]:
    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# PROCESS DATA
# ============================================================

total_source_images = 0
total_output_images = 0
total_boxes = 0
removed_boxes = 0

class_counts = Counter()
split_images = Counter()
split_boxes = Counter()

for split in ["train", "validation", "test"]:

    source_images_dir = SOURCE / "images" / split
    source_labels_dir = SOURCE / "labels" / split

    output_images_dir = OUTPUT / "images" / split
    output_labels_dir = OUTPUT / "labels" / split

    image_files = [
        p for p in source_images_dir.iterdir()
        if p.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]
    ]

    print(f"\nProcessing {split}...")
    print("Source images:", len(image_files))

    for image_file in image_files:

        total_source_images += 1

        label_file = source_labels_dir / f"{image_file.stem}.txt"

        if not label_file.exists():
            raise FileNotFoundError(
                f"Missing label for {image_file}"
            )

        selected_lines = []

        with open(label_file, "r") as f:
            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO line:\n"
                    f"{label_file}\n"
                    f"{line}"
                )

            old_class_id = int(parts[0])

            # ------------------------------------------------
            # Keep only requested 78 classes
            # ------------------------------------------------

            if old_class_id not in id_mapping:

                removed_boxes += 1
                continue

            new_class_id = id_mapping[old_class_id]

            selected_lines.append(
                f"{new_class_id} "
                f"{parts[1]} "
                f"{parts[2]} "
                f"{parts[3]} "
                f"{parts[4]}"
            )

            class_counts[new_class_id] += 1
            split_boxes[split] += 1
            total_boxes += 1

        # ----------------------------------------------------
        # If image contains none of the 78 classes,
        # don't copy it.
        # ----------------------------------------------------

        if not selected_lines:
            continue

        # Copy image
        shutil.copy2(
            image_file,
            output_images_dir / image_file.name
        )

        # Write remapped labels
        output_label = (
            output_labels_dir /
            f"{image_file.stem}.txt"
        )

        with open(output_label, "w") as f:
            f.write("\n".join(selected_lines))
            f.write("\n")

        total_output_images += 1
        split_images[split] += 1

# ============================================================
# CREATE DATA.YAML
# ============================================================

yaml_file = OUTPUT / "data.yaml"

with open(yaml_file, "w") as f:

    f.write(f"path: {OUTPUT}\n")
    f.write("train: images/train\n")
    f.write("val: images/validation\n")
    f.write("test: images/test\n\n")

    f.write("nc: 78\n")
    f.write("names:\n")

    for i, name in enumerate(classes):
        f.write(f"  {i}: {name}\n")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("78-CLASS DATASET COMPLETE")
print("========================================")

print(f"Source images processed : {total_source_images}")
print(f"Output images           : {total_output_images}")
print(f"Output bounding boxes   : {total_boxes}")
print(f"Removed unwanted boxes  : {removed_boxes}")

print("\n--- SPLIT ---")

for split in ["train", "validation", "test"]:
    print(
        f"{split:12s} "
        f"images={split_images[split]:6d} "
        f"boxes={split_boxes[split]:6d}"
    )

print("\n--- CLASS BOX COUNTS ---")

for i, name in enumerate(classes):
    print(
        f"{i:2d} {name:10s}: "
        f"{class_counts[i]:5d}"
    )

print("\nClasses:", len(classes))
print("data.yaml:")
print(yaml_file)

print("\n========================================")
