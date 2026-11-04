import csv
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

ROOT = Path("/home/admin1/Abishek-Intern/new_atr/data/aircrafts/J35")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PART_CLASSES = {"nose", "wings", "tail"}

print("\n========== J35 DATASET AUDIT ==========\n")

# 1. Index images
image_index = defaultdict(list)

for p in ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        image_index[p.stem].append(p)

print("Total images:", sum(len(v) for v in image_index.values()))
print("Unique image names:", len(image_index))

# 2. Find annotation CSVs
csv_files = list(ROOT.rglob("*.csv"))

part_files = [
    p for p in csv_files
    if p.stem.lower().endswith("_parts")
]

main_files = [
    p for p in csv_files
    if not p.stem.lower().endswith("_parts")
]

print("\nTotal CSV files:", len(csv_files))
print("Main annotation CSVs:", len(main_files))
print("Parts annotation CSVs:", len(part_files))

# 3. Store annotations
main_annotations = defaultdict(list)
part_annotations = defaultdict(list)

main_classes = Counter()
part_classes = Counter()

errors = []
dimension_errors = []

def process_csv(path, is_part):

    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)

            required = {
                "filename", "width", "height",
                "xmin", "ymin", "xmax", "ymax"
            }

            if not required.issubset(reader.fieldnames or []):
                errors.append(f"Missing columns: {path.name}")
                return

            for row in reader:

                filename = Path(row["filename"]).stem

                try:
                    w = float(row["width"])
                    h = float(row["height"])
                    xmin = float(row["xmin"])
                    ymin = float(row["ymin"])
                    xmax = float(row["xmax"])
                    ymax = float(row["ymax"])

                except (ValueError, TypeError):
                    errors.append(f"Invalid numeric data: {filename}")
                    continue

                if is_part:
                    part = row.get("part", "").strip().lower()

                    if part not in PART_CLASSES:
                        errors.append(
                            f"Unknown part '{part}' in {filename}"
                        )
                        continue

                    part_annotations[filename].append(row)
                    part_classes[part] += 1

                else:
                    cls = row.get(
                        "class",
                        row.get("aircraft_class", "")
                    ).strip()

                    main_annotations[filename].append(row)
                    main_classes[cls] += 1

                # Check image availability
                matches = image_index.get(filename, [])

                if len(matches) == 0:
                    errors.append(f"Missing image: {filename}")
                    continue

                if len(matches) > 1:
                    errors.append(f"Duplicate image filename: {filename}")
                    continue

                # Check actual image dimensions
                try:
                    with Image.open(matches[0]) as img:
                        actual_w, actual_h = img.size

                    if (actual_w, actual_h) != (w, h):
                        dimension_errors.append(
                            f"{filename}: CSV={w}x{h}, "
                            f"Image={actual_w}x{actual_h}"
                        )

                except Exception as e:
                    errors.append(f"Image read error {filename}: {e}")
                    continue

                # Validate bounding boxes
                if (
                    w <= 0 or h <= 0
                    or xmin < 0 or ymin < 0
                    or xmax > w or ymax > h
                    or xmax <= xmin or ymax <= ymin
                ):
                    errors.append(f"Invalid bounding box: {filename}")

    except Exception as e:
        errors.append(f"CSV read error {path.name}: {e}")


# 4. Process CSV files
for p in main_files:
    process_csv(p, False)

for p in part_files:
    process_csv(p, True)

# 5. Compare annotation coverage
main_images = set(main_annotations)
part_images = set(part_annotations)
all_images = set(image_index)

both = main_images & part_images

print("\n========== ANNOTATION STATISTICS ==========")

print("Images with main annotations:", len(main_images))
print("Images with part annotations:", len(part_images))
print("Images with BOTH annotations:", len(both))

print("\nMain aircraft classes:")
for cls, count in main_classes.items():
    print(f"  {cls}: {count}")

print("\nComponent distribution:")
for cls in ["nose", "wings", "tail"]:
    print(f"  {cls}: {part_classes[cls]}")

# 6. Multiple aircraft
multiple = [
    (name, len(rows))
    for name, rows in main_annotations.items()
    if len(rows) > 1
]

print("\n========== MULTIPLE AIRCRAFT ==========")
print("Images with multiple aircraft:", len(multiple))

for name, count in multiple[:20]:
    print(f"  {name}: {count} aircraft")

# 7. Missing annotations
print("\n========== MISSING ANNOTATIONS ==========")

print("Images without main annotations:",
      len(all_images - main_images))

print("Images without part annotations:",
      len(all_images - part_images))

print("Main annotation images without parts:",
      len(main_images - part_images))

print("Part annotation images without main:",
      len(part_images - main_images))

# 8. Errors
print("\n========== VALIDATION ERRORS ==========")
print("Total errors:", len(errors))

for e in errors[:30]:
    print("  ", e)

if len(errors) > 30:
    print("  Additional errors:", len(errors) - 30)

print("\n========== DIMENSION MISMATCHES ==========")
print("Total mismatches:", len(dimension_errors))

for e in dimension_errors[:20]:
    print("  ", e)

print("\n========================================")
print("J35 DATASET AUDIT COMPLETED")
print("Original dataset NOT modified.")
print("========================================")
