#!/usr/bin/env python3
"""
Convert the user's CSV-annotated aircraft data to a filtered YOLO detection dataset.

Inputs:
  Full images: /home/admin1/Abishek-Intern/new_atr/data/raw/dataset/
  Cropped images: /home/admin1/Abishek-Intern/new_atr/data/raw/crop/
  Master CSV: /home/admin1/Abishek-Intern/new_atr/data/raw/labels_with_split.csv

Output:
  /home/admin1/Abishek-Intern/new_atr/data/selected_79class_yolo/

Only the requested aircraft classes are retained. Existing CSV split values are
preserved. The raw source folders are never modified.
"""

from pathlib import Path
from collections import defaultdict, Counter
import csv
import shutil
import sys
import yaml
import cv2

BASE = Path("/home/admin1/Abishek-Intern/new_atr/data")
RAW = BASE / "raw"
IMAGE_DIRS = [
    ("dataset", RAW / "dataset"),
    ("crop", RAW / "crop"),
]
MASTER_CSV = RAW / "labels_with_split.csv"
OUTPUT = BASE / "selected_79class_yolo"

CLASSES = [
    "A10", "A400M", "AG600", "AH64", "AV8B", "An124", "An22", "An225",
    "An72", "B1", "B2", "B52", "Be200", "C1", "C130", "C17", "C2",
    "C390", "C5", "CH47", "CL415", "E2", "E7", "EF2000", "EMB314",
    "F117", "F14", "F15", "F16", "F18", "F2", "F22", "F35", "F4",
    "H6", "Il76", "J10", "J20", "J35", "JAS39", "JF17", "JH7", "KC135",
    "KF21", "KJ600", "Ka27", "Ka52", "MQ9", "Mi24", "Mi26", "Mi28",
    "Mi8", "Mig29", "Mig31", "Mirage2000", "P3", "RQ4", "Rafale", "SR71",
    "Su24", "Su25", "Su34", "Su47", "Su57", "TB2", "Tornado", "Tu160",
    "Tu22M", "Tu95", "U2", "UH60", "US2", "V22", "Vulcan", "WZ7",
    "XB70", "Y20", "YF23", "Tejas"
]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_MAP = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


def canonical(s):
    return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum())


def get_col(headers, aliases):
    by_norm = {canonical(h): h for h in headers}
    for alias in aliases:
        if canonical(alias) in by_norm:
            return by_norm[canonical(alias)]
    return None


def find_images():
    index = defaultdict(list)
    for source_name, root in IMAGE_DIRS:
        if not root.is_dir():
            print(f"WARNING: image directory not found (skipping): {root}")
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTS:
                item = (source_name, root, path)
                index[path.name.lower()].append(item)
                # The master CSV may store hash filenames without extensions.
                index[path.stem.lower()].append(item)
    return index

def select_image(filename, expected_w, expected_h, index):
    basename = Path(filename.replace("\\\\", "/")).name.lower()
    stem = Path(basename).stem.lower()

    # Support both CSV filenames with extensions and hash filenames without them.
    candidates = list(index.get(basename, []))
    if stem != basename:
        candidates.extend(index.get(stem, []))
    # De-duplicate because each image is indexed by both name and stem.
    unique = {}
    for item in candidates:
        unique[str(item[2].resolve())] = item
    candidates = list(unique.values())

    if not candidates:
        return None, "not found by filename or filename stem"

    exact = []
    for source_name, root, path in candidates:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if abs(w - expected_w) < 1.0 and abs(h - expected_h) < 1.0:
            exact.append((source_name, root, path))

    if not exact:
        dims = []
        for source_name, root, path in candidates:
            img = cv2.imread(str(path))
            if img is not None:
                h, w = img.shape[:2]
                dims.append(f"{path} [{w}x{h}]")
        return None, (
            f"no image with CSV dimensions {expected_w:g}x{expected_h:g}; "
            + "; ".join(dims[:4])
        )

    # Prefer original/full images over crop images when both match.
    exact.sort(key=lambda item: (0 if item[0] == "dataset" else 1, str(item[2])))
    preferred = exact[0]
    same_priority = [x for x in exact if x[0] == preferred[0]]
    if len(same_priority) > 1:
        return None, "duplicate basename with matching dimensions: " + ", ".join(
            str(x[2]) for x in same_priority
        )
    return preferred, "matched"

def main():
    if not MASTER_CSV.is_file():
        sys.exit(f"ERROR: master CSV not found: {MASTER_CSV}")
    if not any(root.is_dir() for _, root in IMAGE_DIRS):
        sys.exit("ERROR: neither raw/dataset nor raw/crop image directory exists")

    class_to_new_id = {canonical(name): i for i, name in enumerate(CLASSES)}
    grouped = defaultdict(list)
    image_meta = {}
    unknown_splits = Counter()
    source_class_counts = Counter()
    selected_class_counts = Counter()

    with MASTER_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        cols = {
            "filename": get_col(headers, ["filename", "image", "image_name", "file"]),
            "width": get_col(headers, ["width", "image_width"]),
            "height": get_col(headers, ["height", "image_height"]),
            "class": get_col(headers, ["class", "class_name", "label", "aircraft_class"]),
            "xmin": get_col(headers, ["xmin", "x_min"]),
            "ymin": get_col(headers, ["ymin", "y_min"]),
            "xmax": get_col(headers, ["xmax", "x_max"]),
            "ymax": get_col(headers, ["ymax", "y_max"]),
            "split": get_col(headers, ["split", "set", "subset"]),
        }
        missing = [k for k, v in cols.items() if v is None]
        if missing:
            sys.exit(f"CSV missing required columns {missing}. Found: {headers}")

        for line_no, row in enumerate(reader, start=2):
            filename = str(row[cols["filename"]]).strip()
            cls = str(row[cols["class"]]).strip()
            source_class_counts[cls] += 1
            new_id = class_to_new_id.get(canonical(cls))
            if new_id is None:
                continue

            split_raw = str(row[cols["split"]]).strip().lower()
            split = SPLIT_MAP.get(split_raw)
            if split is None:
                unknown_splits[split_raw] += 1
                continue

            try:
                w = float(row[cols["width"]])
                h = float(row[cols["height"]])
                x1 = float(row[cols["xmin"]])
                y1 = float(row[cols["ymin"]])
                x2 = float(row[cols["xmax"]])
                y2 = float(row[cols["ymax"]])
            except Exception as exc:
                raise ValueError(f"Invalid numeric value at CSV line {line_no}: {exc}")

            if w <= 0 or h <= 0 or x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid dimensions/bounding box at CSV line {line_no}")

            key = (Path(filename.replace("\\", "/")).name.lower(), split)
            if key in image_meta and image_meta[key] != (filename, w, h):
                old_filename, old_w, old_h = image_meta[key]
                if (old_w, old_h) != (w, h):
                    raise ValueError(
                        f"Same image basename has inconsistent dimensions: "
                        f"{old_filename} vs {filename}"
                    )
            image_meta[key] = (filename, w, h)
            grouped[key].append((new_id, x1, y1, x2, y2))
            selected_class_counts[CLASSES[new_id]] += 1

    if unknown_splits:
        print("WARNING: selected-class annotations with unrecognized split values:")
        for split, n in unknown_splits.items():
            print(f"  {split!r}: {n} rows (not included)")

    if not grouped:
        sys.exit("ERROR: no annotations matched the selected 79 classes and known splits.")

    print("Indexing images in raw/dataset and raw/crop ...")
    image_index = find_images()
    print(f"Indexed image files: {sum(len(v) for v in image_index.values())}")

    resolved = {}
    problems = []
    for key, (filename, w, h) in image_meta.items():
        selected, reason = select_image(filename, w, h, image_index)
        if selected is None:
            problems.append(f"{filename} ({w:g}x{h:g}): {reason}")
        else:
            resolved[key] = selected

    if problems:
        print(f"\nERROR: could not safely resolve {len(problems)} annotated images:")
        for problem in problems[:30]:
            print("  -", problem)
        sys.exit(
            "No output was created. Resolve missing/duplicate images or verify CSV dimensions."
        )

    # Do not remove an existing output folder until input validation succeeds.
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for split in ("train", "val", "test"):
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_image_counts = Counter()
    split_box_counts = Counter()
    class_split_counts = {name: Counter() for name in CLASSES}
    copied_sources = Counter()

    for key, boxes in grouped.items():
        basename, split = key
        source_name, root, image_path = resolved[key]
        rel = image_path.relative_to(root)
        # Prefix source folder to prevent collisions between dataset/ and crop/.
        out_rel = Path(source_name) / rel
        out_img = OUTPUT / "images" / split / out_rel
        out_lbl = (OUTPUT / "labels" / split / out_rel).with_suffix(".txt")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        actual_h, actual_w = img.shape[:2]
        _, expected_w, expected_h = image_meta[key]
        if actual_w != round(expected_w) or actual_h != round(expected_h):
            raise RuntimeError(
                f"Dimension mismatch after resolving {image_path}: "
                f"actual {actual_w}x{actual_h}, CSV {expected_w}x{expected_h}"
            )

        yolo_lines = []
        for class_id, x1, y1, x2, y2 in boxes:
            # Clip boxes to image edges, then normalize.
            x1 = max(0.0, min(float(actual_w), x1))
            x2 = max(0.0, min(float(actual_w), x2))
            y1 = max(0.0, min(float(actual_h), y1))
            y2 = max(0.0, min(float(actual_h), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2.0) / actual_w
            cy = ((y1 + y2) / 2.0) / actual_h
            bw = (x2 - x1) / actual_w
            bh = (y2 - y1) / actual_h
            yolo_lines.append(
                f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"
            )
            split_box_counts[split] += 1
            class_split_counts[CLASSES[class_id]][split] += 1

        if not yolo_lines:
            continue
        shutil.copy2(image_path, out_img)
        out_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        split_image_counts[split] += 1
        copied_sources[source_name] += 1

    yaml_obj = {
        "path": str(OUTPUT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    (OUTPUT / "data.yaml").write_text(
        yaml.safe_dump(yaml_obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    report = [
        f"Master CSV: {MASTER_CSV}",
        f"Image sources: {', '.join(str(p) for _, p in IMAGE_DIRS)}",
        f"Output: {OUTPUT}",
        f"Requested classes: {len(CLASSES)}",
        f"Images sourced from: {dict(copied_sources)}",
        "",
        "SPLIT SUMMARY",
    ]
    for split in ("train", "val", "test"):
        report.append(
            f"{split}: images={split_image_counts[split]}, "
            f"bounding_boxes={split_box_counts[split]}"
        )
    report.append(
        f"TOTAL: images={sum(split_image_counts.values())}, "
        f"bounding_boxes={sum(split_box_counts.values())}"
    )
    report += ["", "CLASS-WISE BOX COUNTS"]
    for name in CLASSES:
        c = class_split_counts[name]
        report.append(
            f"{name:12s} train={c['train']:7d} val={c['val']:7d} "
            f"test={c['test']:7d} total={sum(c.values()):7d}"
        )
    report_path = OUTPUT / "filter_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nYOLO CONVERSION COMPLETE")
    print("Output:", OUTPUT)
    for split in ("train", "val", "test"):
        print(
            f"{split:5s}: {split_image_counts[split]} images, "
            f"{split_box_counts[split]} boxes"
        )
    print(
        f"TOTAL: {sum(split_image_counts.values())} images, "
        f"{sum(split_box_counts.values())} boxes"
    )
    print("Image sources used:", dict(copied_sources))
    print("YAML:", OUTPUT / "data.yaml")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
