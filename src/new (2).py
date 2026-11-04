#!/usr/bin/env python3
"""
Build two YOLO datasets from the J35 source folder:

1) j35_aircraft_detection: one class (J35), full original images.
2) j35_part_detection: three classes (nose, wings, tail), one padded crop
   per aircraft instance, with part boxes converted to crop coordinates.

Expected source layout:
    J35/
      image_001.jpg
      image_001.csv
      image_001_parts.csv
      ...

Expected whole-aircraft CSV columns:
    filename,width,height,class,xmin,ymin,xmax,ymax

Expected part CSV columns:
    filename,width,height,aircraft_class,part,xmin,ymin,xmax,ymax

The script does not modify the source folder. It creates datasets under DATA_ROOT.
"""

from __future__ import annotations

import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

# ---------------------------- CONFIGURATION ----------------------------

SOURCE_DIR = Path("/home/admin1/Abishek-Intern/new_atr/data/aircrafts/J35")
DATA_ROOT = Path("/home/admin1/Abishek-Intern/new_atr/data")

AIRCRAFT_OUT = DATA_ROOT / "j35_aircraft_detection"
PART_OUT = DATA_ROOT / "j35_part_detection"
REPORT_DIR = DATA_ROOT / "j35_dataset_build_report"

SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
RANDOM_SEED = 42

# Padding around the whole-aircraft box before making a part crop.
CROP_PADDING_FRACTION = 0.05

# A part box is considered a candidate for an aircraft if its centre is
# inside that aircraft box. If its centre is outside all aircraft boxes,
# substantial overlap can still be used as a fallback.
FALLBACK_PART_OVERLAP = 0.50

# If the best two aircraft candidates have very similar scores, do not guess.
AMBIGUITY_MARGIN = 0.10

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PART_CLASS_TO_ID = {"nose": 0, "wings": 1, "tail": 2}


# ------------------------------- HELPERS -------------------------------

def norm_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(fieldnames, candidates):
    lookup = {norm_name(x): x for x in (fieldnames or [])}
    for candidate in candidates:
        key = norm_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


def read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], {}
        fields = reader.fieldnames
        cols = {
            "filename": find_column(fields, ["filename", "file", "image", "image_name"]),
            "width": find_column(fields, ["width", "image_width"]),
            "height": find_column(fields, ["height", "image_height"]),
            "class": find_column(fields, ["class", "aircraft_class", "label", "part"]),
            "part": find_column(fields, ["part", "class", "label"]),
            "xmin": find_column(fields, ["xmin", "x_min", "left"]),
            "ymin": find_column(fields, ["ymin", "y_min", "top"]),
            "xmax": find_column(fields, ["xmax", "x_max", "right"]),
            "ymax": find_column(fields, ["ymax", "y_max", "bottom"]),
        }
        return list(reader), cols


def to_float(row, key, cols):
    col = cols.get(key)
    if not col or row.get(col) in (None, ""):
        raise ValueError(f"Missing value for {key}")
    return float(row[col])


def parse_boxes(csv_path: Path, is_parts: bool):
    rows, cols = read_csv_rows(csv_path)
    required = ["xmin", "ymin", "xmax", "ymax"]
    missing = [x for x in required if not cols.get(x)]
    if missing:
        raise ValueError(f"{csv_path.name}: missing coordinate columns {missing}")

    boxes = []
    for line_num, row in enumerate(rows, start=2):
        try:
            x1 = to_float(row, "xmin", cols)
            y1 = to_float(row, "ymin", cols)
            x2 = to_float(row, "xmax", cols)
            y2 = to_float(row, "ymax", cols)
            label_col = cols["part"] if is_parts else cols["class"]
            label = norm_name(row.get(label_col, "")) if label_col else ""
            if x2 <= x1 or y2 <= y1:
                raise ValueError("non-positive box width/height")
            if is_parts:
                # Normalize common variants.
                if label in {"wing", "wings"}:
                    label = "wings"
                elif label in {"nose", "nose_cone"}:
                    label = "nose"
                elif label in {"tail", "empennage"}:
                    label = "tail"
                if label not in PART_CLASS_TO_ID:
                    raise ValueError(f"unknown part label '{label}'")
            boxes.append({
                "label": label,
                "xyxy": [x1, y1, x2, y2],
                "line": line_num,
            })
        except Exception as exc:
            raise ValueError(f"{csv_path.name}, line {line_num}: {exc}") from exc
    return boxes


def yolo_line(class_id, box, image_w, image_h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(image_w), x1))
    x2 = max(0.0, min(float(image_w), x2))
    y1 = max(0.0, min(float(image_h), y1))
    y2 = max(0.0, min(float(image_h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2.0) / image_w
    cy = ((y1 + y2) / 2.0) / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h
    return f"{class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def assign_part_to_aircraft(part_box, aircraft_boxes):
    """Return (aircraft_index, reason, ambiguous). Never silently guess."""
    px1, py1, px2, py2 = part_box
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2

    containing = []
    for idx, aircraft in enumerate(aircraft_boxes):
        ax1, ay1, ax2, ay2 = aircraft
        if ax1 <= pcx <= ax2 and ay1 <= pcy <= ay2:
            # Prefer the smallest containing aircraft box if boxes overlap.
            containing.append((box_area(aircraft), idx))
    if containing:
        containing.sort()
        if len(containing) > 1:
            first, second = containing[0], containing[1]
            if second[0] > 0 and (second[0] - first[0]) / second[0] < AMBIGUITY_MARGIN:
                return None, "ambiguous overlapping aircraft boxes", True
        return containing[0][1], "part centre inside aircraft box", False

    part_area = box_area(part_box)
    scores = []
    for idx, aircraft in enumerate(aircraft_boxes):
        overlap = intersection_area(part_box, aircraft)
        ratio = overlap / part_area if part_area else 0
        if ratio >= FALLBACK_PART_OVERLAP:
            scores.append((ratio, idx))
    scores.sort(reverse=True)
    if not scores:
        return None, "no aircraft box contains/overlaps part sufficiently", True
    if len(scores) > 1 and scores[0][0] - scores[1][0] < AMBIGUITY_MARGIN:
        return None, "ambiguous part overlap with multiple aircraft", True
    return scores[0][1], "fallback overlap assignment", False


def split_images(images):
    rng = random.Random(RANDOM_SEED)
    images = list(images)
    rng.shuffle(images)
    n = len(images)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    # Put rounding remainder in test, ensuring every image is assigned once.
    return {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }


def prepare_dirs():
    for root in (AIRCRAFT_OUT, PART_OUT):
        if root.exists():
            shutil.rmtree(root)
        for split in SPLIT_RATIOS:
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def write_yaml(path, names):
    lines = [
        f"path: {path.parent}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    (path / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------- MAIN ----------------------------------

def main():
    if not SOURCE_DIR.is_dir():
        sys.exit(f"ERROR: source folder not found: {SOURCE_DIR}")

    images = sorted(
        p for p in SOURCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        sys.exit(f"ERROR: no images found directly inside {SOURCE_DIR}")

    # Validate required CSV pairing and parse annotations before deleting outputs.
    records = []
    missing_whole = []
    bad_records = []
    for image_path in images:
        whole_csv = image_path.with_suffix(".csv")
        parts_csv = image_path.with_name(image_path.stem + "_parts.csv")
        if not whole_csv.exists():
            missing_whole.append(image_path.name)
            continue
        try:
            aircraft = parse_boxes(whole_csv, is_parts=False)
            if not aircraft:
                raise ValueError("whole-aircraft CSV has no boxes")
            parts = parse_boxes(parts_csv, is_parts=True) if parts_csv.exists() else None
            records.append({
                "image": image_path,
                "whole_csv": whole_csv,
                "parts_csv": parts_csv if parts_csv.exists() else None,
                "aircraft": aircraft,
                "parts": parts,
            })
        except Exception as exc:
            bad_records.append(f"{image_path.name}: {exc}")

    if missing_whole or bad_records:
        print("ERROR: source validation found issues. No output folders were changed.")
        for name in missing_whole[:30]:
            print("  Missing whole CSV:", name)
        for msg in bad_records[:30]:
            print("  Invalid annotation:", msg)
        print("Fix these issues and rerun. Full list is written only after successful validation.")
        sys.exit(1)

    splits = split_images([r["image"].name for r in records])
    split_for_name = {name: split for split, names in splits.items() for name in names}

    print(f"Validated images: {len(records)}")
    print("Image-level split:", {k: len(v) for k, v in splits.items()})
    print("Preparing output folders (existing generated output folders will be replaced)...")
    prepare_dirs()

    report_rows = []
    assignment_rows = []
    totals = Counter()
    part_crop_count = Counter()
    skipped_part_crops = Counter()

    record_by_name = {r["image"].name: r for r in records}

    for split, names in splits.items():
        for image_name in names:
            rec = record_by_name[image_name]
            image_path = rec["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                report_rows.append(f"IMAGE_READ_ERROR,{image_name}")
                continue
            height, width = image.shape[:2]
            aircraft_boxes = [a["xyxy"] for a in rec["aircraft"]]

            # Model 1: original image with all J35 aircraft boxes.
            shutil.copy2(image_path, AIRCRAFT_OUT / "images" / split / image_path.name)
            det_lines = []
            for a in rec["aircraft"]:
                line = yolo_line(0, a["xyxy"], width, height)
                if line:
                    det_lines.append(line)
                    totals[f"aircraft_{split}"] += 1
            det_label = AIRCRAFT_OUT / "labels" / split / f"{image_path.stem}.txt"
            det_label.write_text("\n".join(det_lines) + ("\n" if det_lines else ""), encoding="utf-8")

            # Model 2: only use crops whose source image has a parts CSV.
            if rec["parts"] is None:
                skipped_part_crops["missing_parts_csv"] += len(aircraft_boxes)
                report_rows.append(f"MISSING_PARTS_CSV,{image_name},all aircraft crops excluded")
                continue

            per_aircraft_parts = defaultdict(list)
            ambiguous_aircraft = set()
            for part in rec["parts"]:
                assigned, reason, ambiguous = assign_part_to_aircraft(
                    part["xyxy"], aircraft_boxes
                )
                assignment_rows.append(
                    f"{image_name},{part['line']},{part['label']},"
                    f"{assigned if assigned is not None else 'UNASSIGNED'},{reason}"
                )
                if ambiguous or assigned is None:
                    # Exclude every aircraft box that the part could plausibly affect:
                    # safest fallback is to exclude this source image's part crops.
                    ambiguous_aircraft.update(range(len(aircraft_boxes)))
                    report_rows.append(
                        f"AMBIGUOUS_PART_ASSIGNMENT,{image_name},CSV line {part['line']},"
                        f"{part['label']},{reason}; all crops from this image excluded"
                    )
                else:
                    per_aircraft_parts[assigned].append(part)

            if ambiguous_aircraft:
                skipped_part_crops["ambiguous_assignment_image"] += len(aircraft_boxes)
                continue

            for aircraft_idx, aircraft in enumerate(aircraft_boxes):
                x1, y1, x2, y2 = aircraft
                pad_x = (x2 - x1) * CROP_PADDING_FRACTION
                pad_y = (y2 - y1) * CROP_PADDING_FRACTION
                cx1 = max(0, int(round(x1 - pad_x)))
                cy1 = max(0, int(round(y1 - pad_y)))
                cx2 = min(width, int(round(x2 + pad_x)))
                cy2 = min(height, int(round(y2 + pad_y)))
                if cx2 <= cx1 or cy2 <= cy1:
                    report_rows.append(f"INVALID_CROP,{image_name},aircraft_index={aircraft_idx}")
                    skipped_part_crops["invalid_crop"] += 1
                    continue

                crop = image[cy1:cy2, cx1:cx2]
                crop_h, crop_w = crop.shape[:2]
                crop_name = f"{image_path.stem}__aircraft_{aircraft_idx:03d}.jpg"
                image_dest = PART_OUT / "images" / split / crop_name
                label_dest = PART_OUT / "labels" / split / f"{Path(crop_name).stem}.txt"
                if not cv2.imwrite(str(image_dest), crop):
                    report_rows.append(f"CROP_WRITE_ERROR,{image_name},{crop_name}")
                    skipped_part_crops["write_error"] += 1
                    continue

                part_lines = []
                for part in per_aircraft_parts.get(aircraft_idx, []):
                    px1, py1, px2, py2 = part["xyxy"]
                    local_box = [px1 - cx1, py1 - cy1, px2 - cx1, py2 - cy1]
                    line = yolo_line(
                        PART_CLASS_TO_ID[part["label"]], local_box, crop_w, crop_h
                    )
                    if line:
                        part_lines.append(line)
                        totals[f"part_{split}_{part['label']}"] += 1
                label_dest.write_text(
                    "\n".join(part_lines) + ("\n" if part_lines else ""),
                    encoding="utf-8",
                )
                part_crop_count[split] += 1

    write_yaml(AIRCRAFT_OUT, ["J35"])
    write_yaml(PART_OUT, ["nose", "wings", "tail"])

    (REPORT_DIR / "build_report.txt").write_text(
        "\n".join([
            f"Source: {SOURCE_DIR}",
            f"Images validated: {len(records)}",
            f"Split counts: {dict((k, len(v)) for k, v in splits.items())}",
            f"Aircraft box counts: {dict((k, totals[f'aircraft_{k}']) for k in SPLIT_RATIOS)}",
            f"Part crop counts: {dict(part_crop_count)}",
            f"Skipped part crops: {dict(skipped_part_crops)}",
            f"Part box counts: {dict(totals)}",
            "Note: images without a parts CSV are excluded from part training, not treated as confirmed absence of parts.",
            "Note: ambiguous part-to-aircraft assignments cause all part crops from that source image to be excluded.",
        ]) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "issues.csv").write_text(
        "issue\n" + "\n".join(report_rows) + ("\n" if report_rows else ""),
        encoding="utf-8",
    )
    (REPORT_DIR / "part_assignment_audit.csv").write_text(
        "image,parts_csv_line,part,aircraft_index,assignment_reason\n"
        + "\n".join(assignment_rows) + ("\n" if assignment_rows else ""),
        encoding="utf-8",
    )

    print("\nDATASET GENERATION COMPLETE")
    print("Aircraft dataset:", AIRCRAFT_OUT)
    print("Part dataset:", PART_OUT)
    print("Reports:", REPORT_DIR)
    print("\nAircraft boxes:", {k: totals[f"aircraft_{k}"] for k in SPLIT_RATIOS})
    print("Part crops:", dict(part_crop_count))
    print("Skipped part crops:", dict(skipped_part_crops))
    print("\nIMPORTANT: inspect issues.csv and part_assignment_audit.csv before training.")


if __name__ == "__main__":
    main()
