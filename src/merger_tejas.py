from pathlib import Path
import hashlib
import shutil
import yaml

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path("/home/admin1/Abishek-Intern/new_atr")

SOURCE = PROJECT / "data/tejas_only"
TARGET = PROJECT / "data/yolo_78class_balanced_aug"

SOURCE_YAML = SOURCE / "data.yaml"
TARGET_YAML = TARGET / "data.yaml"

TEJAS_ID = 78

# First run with True to preview the merge.
# After reviewing the output, change to False.
DRY_RUN = False

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

SPLITS = ("train", "validation", "test")


# ============================================================
# HELPERS
# ============================================================

def sha256(path):
    """Calculate an image's SHA-256 hash."""
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True
        )


def get_names(data):
    names = data.get("names")

    if isinstance(names, dict):
        names = [
            names[i] if i in names else names[str(i)]
            for i in range(len(names))
        ]

    return list(names)


def get_images(folder):
    if not folder.exists():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_tejas_labels(label_path):
    """
    Read YOLO labels from the one-class Tejas dataset.
    Convert class ID 0 to class ID 78.
    """
    if not label_path.exists():
        raise ValueError(f"Missing label file: {label_path}")

    converted = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty label file: {label_path}")

    for line_number, line in enumerate(lines, start=1):
        parts = line.strip().split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO label in {label_path}, "
                f"line {line_number}"
            )

        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            raise ValueError(
                f"Non-numeric label in {label_path}, "
                f"line {line_number}"
            )

        if class_id != 0:
            raise ValueError(
                f"Expected Tejas source class ID 0, "
                f"found {class_id} in {label_path}"
            )

        # YOLO coordinates must be normalized.
        if not all(0 <= value <= 1 for value in coords):
            raise ValueError(
                f"Coordinates outside [0, 1] in {label_path}, "
                f"line {line_number}"
            )

        converted.append(
            "78 " + " ".join(parts[1:])
        )

    return converted


def read_existing_label_lines(label_path):
    if not label_path.exists():
        return []

    with open(label_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def write_labels(label_path, lines):
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


# ============================================================
# VALIDATE DATASETS
# ============================================================

print("=" * 65)
print("TEJAS -> 78-CLASS DATASET MERGE")
print("=" * 65)

if not SOURCE_YAML.exists():
    raise FileNotFoundError(f"Source YAML not found: {SOURCE_YAML}")

if not TARGET_YAML.exists():
    raise FileNotFoundError(f"Target YAML not found: {TARGET_YAML}")

source_config = load_yaml(SOURCE_YAML)
target_config = load_yaml(TARGET_YAML)

source_names = get_names(source_config)
target_names = get_names(target_config)

if source_config.get("nc") != 1 or source_names != ["Tejas"]:
    raise ValueError(
        "The source dataset must contain exactly one class: Tejas."
    )

if target_config.get("nc") == 78:
    if len(target_names) != 78:
        raise ValueError(
            "Target YAML says nc=78 but its class list "
            "does not contain exactly 78 classes."
        )

    if "Tejas" in target_names:
        raise ValueError(
            "Tejas already appears in the class list. "
            "Check the target YAML before proceeding."
        )

    updated_names = target_names + ["Tejas"]

elif (
    target_config.get("nc") == 79
    and len(target_names) == 79
    and target_names[78] == "Tejas"
):
    # Supports safely rerunning an already completed merge.
    updated_names = target_names

else:
    raise ValueError(
        "Unexpected target class configuration. "
        "Expected 78 classes, or 79 classes with Tejas at ID 78."
    )

if target_config.get("path") != str(TARGET):
    raise ValueError(
        f"Unexpected target path in YAML: {target_config.get('path')}"
    )

for split in SPLITS:
    if not (SOURCE / "images" / split).is_dir():
        raise FileNotFoundError(
            f"Missing source image directory: "
            f"{SOURCE / 'images' / split}"
        )

    if not (SOURCE / "labels" / split).is_dir():
        raise FileNotFoundError(
            f"Missing source label directory: "
            f"{SOURCE / 'labels' / split}"
        )

    if not (TARGET / "images" / split).is_dir():
        raise FileNotFoundError(
            f"Missing target image directory: "
            f"{TARGET / 'images' / split}"
        )

    if not (TARGET / "labels" / split).is_dir():
        raise FileNotFoundError(
            f"Missing target label directory: "
            f"{TARGET / 'labels' / split}"
        )

print(f"\nSource: {SOURCE}")
print(f"Target: {TARGET}")
print(f"Current target classes: {target_config['nc']}")
print(f"Final target classes: {len(updated_names)}")
print(f"Tejas class ID: {TEJAS_ID}")
print(f"DRY_RUN: {DRY_RUN}")


# ============================================================
# INDEX EXISTING TARGET IMAGES BY HASH
# ============================================================

# Maps each image hash to its target split and file path.
target_hashes = {}

for split in SPLITS:
    folder = TARGET / "images" / split

    for image_path in get_images(folder):
        image_hash = sha256(image_path)

        target_hashes.setdefault(image_hash, []).append(
            (split, image_path)
        )


# ============================================================
# PLAN AND PERFORM MERGE
# ============================================================

stats = {
    split: {
        "source_images": 0,
        "copied": 0,
        "duplicate_annotations_merged": 0,
        "duplicate_lines_ignored": 0,
        "conflicts": 0,
    }
    for split in SPLITS
}

planned_operations = []
used_names = {
    split: {
        p.name for p in get_images(TARGET / "images" / split)
    }
    for split in SPLITS
}

for split in SPLITS:
    print(f"\n--- {split.upper()} ---")

    source_image_dir = SOURCE / "images" / split
    source_label_dir = SOURCE / "labels" / split

    target_image_dir = TARGET / "images" / split
    target_label_dir = TARGET / "labels" / split

    images = get_images(source_image_dir)
    stats[split]["source_images"] = len(images)

    for image_path in images:
        label_path = source_label_dir / f"{image_path.stem}.txt"

        # Validate and convert the source label.
        tejas_lines = read_tejas_labels(label_path)
        image_hash = sha256(image_path)

        # Check whether identical image content already exists.
        matches = target_hashes.get(image_hash, [])

        if matches:
            matching_splits = {item[0] for item in matches}

            # Identical content in another split could cause leakage.
            if matching_splits != {split}:
                stats[split]["conflicts"] += 1
                print(
                    f"CONFLICT: {image_path.name} "
                    f"already exists in a different split: "
                    f"{sorted(matching_splits)}. Skipping."
                )
                continue

            # Identical image already exists in this split.
            # Add its Tejas annotations to the existing label.
            existing_image = matches[0][1]
            existing_label = (
                TARGET / "labels" / split /
                f"{existing_image.stem}.txt"
            )

            existing_lines = read_existing_label_lines(existing_label)
            new_lines = [
                line for line in tejas_lines
                if line not in existing_lines
            ]

            if new_lines:
                planned_operations.append(
                    ("merge_labels", existing_label, new_lines)
                )
                stats[split]["duplicate_annotations_merged"] += 1

                print(
                    f"ANNOTATION MERGE: {image_path.name} "
                    f"-> {existing_image.name}; "
                    f"{len(new_lines)} new Tejas boxes"
                )
            else:
                stats[split]["duplicate_lines_ignored"] += 1
                print(
                    f"DUPLICATE: {image_path.name}; "
                    "Tejas annotations already present"
                )

            continue

        # Prefix filenames to avoid collisions with existing files.
        candidate_name = f"tejas_{image_path.name}"
        candidate_stem = Path(candidate_name).stem
        candidate_suffix = Path(candidate_name).suffix

        counter = 1

        while candidate_name in used_names[split]:
            candidate_name = (
                f"tejas_{image_path.stem}_{counter}"
                f"{candidate_suffix}"
            )
            counter += 1

        destination_image = target_image_dir / candidate_name
        destination_label = (
            target_label_dir / f"{Path(candidate_name).stem}.txt"
        )

        planned_operations.append(
            (
                "copy_image_and_labels",
                image_path,
                destination_image,
                destination_label,
                tejas_lines,
            )
        )

        used_names[split].add(candidate_name)

        # Reserve the image hash so repeated source images in this
        # run are checked against the split in which they were placed.
        target_hashes.setdefault(image_hash, []).append(
            (split, destination_image)
        )

        stats[split]["copied"] += 1

        print(
            f"COPY: {image_path.name} -> {candidate_name}; "
            f"{len(tejas_lines)} boxes, class ID 78"
        )


# ============================================================
# APPLY OPERATIONS
# ============================================================

if DRY_RUN:
    print("\nDRY RUN ONLY: no images, labels, or YAML were modified.")
    print("Review the report before enabling the actual merge.")

else:
    # Back up the original target YAML before changing it.
    backup_path = TARGET / "data.yaml.before_tejas_merge.bak"

    if not backup_path.exists():
        shutil.copy2(TARGET_YAML, backup_path)
        print(f"\nYAML backup created: {backup_path}")

    for operation in planned_operations:
        if operation[0] == "copy_image_and_labels":
            _, src_image, dst_image, dst_label, labels = operation

            shutil.copy2(src_image, dst_image)
            write_labels(dst_label, labels)

        elif operation[0] == "merge_labels":
            _, label_path, new_lines = operation

            current_lines = read_existing_label_lines(label_path)
            combined_lines = current_lines + [
                line for line in new_lines
                if line not in current_lines
            ]

            write_labels(label_path, combined_lines)

    # Update class count and names while preserving the existing order.
    target_config["nc"] = 79
    target_config["names"] = updated_names
    save_yaml(target_config, TARGET_YAML)

    print("\nMerge operations applied.")
    print(f"Updated YAML: {TARGET_YAML}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("MERGE SUMMARY")
print("=" * 65)

for split in SPLITS:
    s = stats[split]

    print(f"\n{split.upper()}")
    print(f"  Source images checked:        {s['source_images']}")
    print(f"  New images planned/copied:     {s['copied']}")
    print(f"  Duplicate annotations merged:  {s['duplicate_annotations_merged']}")
    print(f"  Duplicate lines ignored:      {s['duplicate_lines_ignored']}")
    print(f"  Cross-split conflicts:         {s['conflicts']}")

print("\nExisting class IDs 0-77 are preserved.")
print("Tejas uses class ID 78.")
print("Source dataset remains unchanged.")
print("=" * 65)
