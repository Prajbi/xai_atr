from pathlib import Path
from collections import defaultdict, Counter

DATASET = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/balanced_700"
)

CLASS_NAMES = [
    "A10",
    "A400M",
    "B1",
    "B2",
    "B52",
    "C130",
    "C17",
    "C2",
    "EF2000",
    "F15",
    "F16",
    "F18",
    "F22",
    "F35",
    "F4",
    "J20",
    "JAS39",
    "Rafale",
    "Su34",
    "US2",
    "V22",
]

LABEL_DIR = DATASET / "labels" / "train"


# ------------------------------------------------------------
# COUNT UNIQUE IMAGES + INSTANCES
# ------------------------------------------------------------

image_counts = defaultdict(set)
instance_counts = Counter()

label_files = list(LABEL_DIR.glob("*.txt"))

for label_file in label_files:

    classes_in_image = set()

    for line in label_file.read_text().splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        if 0 <= class_id < len(CLASS_NAMES):

            instance_counts[class_id] += 1
            classes_in_image.add(class_id)

    for class_id in classes_in_image:
        image_counts[class_id].add(label_file.stem)


# ------------------------------------------------------------
# PRINT
# ------------------------------------------------------------

print("\n" + "=" * 85)
print("BALANCED 700 — FINAL TRAINING DISTRIBUTION")
print("=" * 85)

print(
    f"{'ID':>3} "
    f"{'CLASS':<10} "
    f"{'IMAGES':>10} "
    f"{'INSTANCES':>12} "
    f"{'AVG OBJ/IMG':>12}"
)

print("-" * 85)

for class_id, name in enumerate(CLASS_NAMES):

    images = len(image_counts[class_id])
    instances = instance_counts[class_id]

    avg = instances / images if images else 0

    print(
        f"{class_id:>3} "
        f"{name:<10} "
        f"{images:>10} "
        f"{instances:>12} "
        f"{avg:>12.2f}"
    )

print("-" * 85)

print(
    f"{'':>3} "
    f"{'TOTAL':<10} "
    f"{len(label_files):>10} "
    f"{sum(instance_counts.values()):>12}"
)

print("\nDataset:")
print(DATASET)

print("\nLabel directory:")
print(LABEL_DIR)
