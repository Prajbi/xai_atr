from pathlib import Path
from collections import Counter
import yaml

BASE = Path("/home/admin1/Abishek-Intern/new_atr/data/yolo_78class")

DATA_YAML = BASE / "data.yaml"
LABEL_DIR = BASE / "labels" / "train"

# Load class names
with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

names = data["names"]

# Handle either list or dictionary format
if isinstance(names, dict):
    names = [names[i] for i in range(len(names))]

class_images = Counter()
class_instances = Counter()

total_images = 0
total_boxes = 0

for label_file in LABEL_DIR.glob("*.txt"):

    total_images += 1
    classes_in_image = set()

    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            class_instances[class_id] += 1
            classes_in_image.add(class_id)
            total_boxes += 1

    for class_id in classes_in_image:
        class_images[class_id] += 1


print("\n" + "=" * 90)
print("78-CLASS TRAINING DATA DISTRIBUTION")
print("=" * 90)

print(f"\nTotal training images : {total_images}")
print(f"Total training boxes  : {total_boxes}")
print(f"Total classes         : {len(names)}")

print("\n" + "-" * 90)
print(
    f"{'ID':>3}  "
    f"{'CLASS':<15} "
    f"{'IMAGES':>8} "
    f"{'INSTANCES':>10} "
    f"{'IMG %':>8} "
    f"{'BOX %':>8}"
)
print("-" * 90)

for class_id, class_name in enumerate(names):

    img_count = class_images[class_id]
    box_count = class_instances[class_id]

    img_pct = (img_count / total_images * 100) if total_images else 0
    box_pct = (box_count / total_boxes * 100) if total_boxes else 0

    print(
        f"{class_id:>3}  "
        f"{class_name:<15} "
        f"{img_count:>8} "
        f"{box_count:>10} "
        f"{img_pct:>7.2f}% "
        f"{box_pct:>7.2f}%"
    )

print("-" * 90)

# Sort by image count
print("\nCLASSES SORTED BY TRAINING IMAGE COUNT")
print("-" * 60)

sorted_classes = sorted(
    range(len(names)),
    key=lambda x: class_images[x]
)

for rank, class_id in enumerate(sorted_classes, 1):
    print(
        f"{rank:>2}. "
        f"{names[class_id]:<15} "
        f"{class_images[class_id]:>6} images  "
        f"{class_instances[class_id]:>6} instances"
    )

print("\n" + "=" * 90)
print("MIN / MAX")
print("=" * 90)

nonzero_images = [class_images[i] for i in range(len(names)) if class_images[i] > 0]

if nonzero_images:
    min_id = min(
        range(len(names)),
        key=lambda x: class_images[x] if class_images[x] > 0 else float("inf")
    )

    max_id = max(
        range(len(names)),
        key=lambda x: class_images[x]
    )

    print(
        f"Minimum : {names[min_id]} "
        f"-> {class_images[min_id]} images"
    )

    print(
        f"Maximum : {names[max_id]} "
        f"-> {class_images[max_id]} images"
    )

    print(
        f"Imbalance ratio : "
        f"{class_images[max_id] / class_images[min_id]:.2f} : 1"
    )

print("=" * 90)
