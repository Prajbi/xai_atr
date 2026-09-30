from pathlib import Path
import pandas as pd
import shutil

# ============================================================
# PATHS
# ============================================================

BASE = Path("/home/admin1/Abishek-Intern/new_atr")

SOURCE = BASE / "data/merged"
IMAGES_SOURCE = SOURCE / "images"
MASTER_CSV = SOURCE / "labels_with_split.csv"

OUTPUT = BASE / "data/yolo_merged"

# ============================================================
# ORIGINAL 102-CLASS MAPPING
# ============================================================

classes = [
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

class_to_id = {
    name: i for i, name in enumerate(classes)
}

print("Number of classes:", len(classes))

# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in ["train", "validation", "test"]:
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD MASTER CSV
# ============================================================

df = pd.read_csv(MASTER_CSV)

print("\nTotal annotation rows:", len(df))
print("Unique images:", df["filename"].nunique())

# ============================================================
# VALIDATE CLASSES
# ============================================================

unknown_classes = set(df["class"]) - set(classes)

if unknown_classes:
    raise ValueError(
        f"Unknown classes found: {unknown_classes}"
    )

# ============================================================
# CONVERT
# ============================================================

total_images = 0
total_labels = 0

for split in ["train", "validation", "test"]:

    split_df = df[df["split"] == split]

    print(f"\nProcessing {split}...")
    print("Images:", split_df["filename"].nunique())
    print("Boxes :", len(split_df))

    for filename, group in split_df.groupby("filename"):

        # ----------------------------------------------------
        # Find source image
        # ----------------------------------------------------

        source_image = None

        for ext in [".jpg", ".jpeg", ".png", ".webp"]:

            candidate = IMAGES_SOURCE / f"{filename}{ext}"

            if candidate.exists():
                source_image = candidate
                break

        if source_image is None:

            raise FileNotFoundError(
                f"Image not found for: {filename}"
            )

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = (
            OUTPUT / "images" / split / source_image.name
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        # ----------------------------------------------------
        # YOLO label file
        # ----------------------------------------------------

        label_file = (
            OUTPUT /
            "labels" /
            split /
            f"{filename}.txt"
        )

        with open(label_file, "w") as f:

            for _, row in group.iterrows():

                class_id = class_to_id[row["class"]]

                img_w = float(row["width"])
                img_h = float(row["height"])

                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])

                # --------------------------------------------
                # Convert XYXY → YOLO XYWH
                # --------------------------------------------

                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h

                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h

                # --------------------------------------------
                # Final safety check
                # --------------------------------------------

                if not (
                    0 <= x_center <= 1 and
                    0 <= y_center <= 1 and
                    0 < width <= 1 and
                    0 < height <= 1
                ):
                    raise ValueError(
                        f"Invalid YOLO box in {filename}: "
                        f"{x_center} {y_center} "
                        f"{width} {height}"
                    )

                f.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

                total_labels += 1

        total_images += 1

# ============================================================
# CREATE DATA.YAML
# ============================================================

yaml_file = OUTPUT / "data.yaml"

with open(yaml_file, "w") as f:

    f.write(f"path: {OUTPUT}\n")
    f.write("train: images/train\n")
    f.write("val: images/validation\n")
    f.write("test: images/test\n\n")
    f.write(f"nc: {len(classes)}\n")
    f.write("names:\n")

    for i, name in enumerate(classes):
        f.write(f"  {i}: {name}\n")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("YOLO CONVERSION COMPLETE")
print("========================================")

print("Total images :", total_images)
print("Total labels :", total_labels)

for split in ["train", "validation", "test"]:

    image_count = len(list(
        (OUTPUT / "images" / split).glob("*")
    ))

    label_count = len(list(
        (OUTPUT / "labels" / split).glob("*.txt")
    ))

    print(
        f"{split:12s} "
        f"images={image_count:6d} "
        f"labels={label_count:6d}"
    )

print("\nClasses:", len(classes))
print("data.yaml:", yaml_file)

print("\n========================================")
