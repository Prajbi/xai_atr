import cv2
import pandas as pd
import random

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/raw/balanced_dataset"
)

IMAGE_DIR = BASE / "images"

CSV_PATH = (
    BASE /
    "labels_with_split_balanced.csv"
)

OUTPUT_DIR = BASE / "visualization"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

df["filename"] = df["filename"].astype(str)
df["class"] = df["class"].astype(str)


# ============================================================
# SELECT AUGMENTED IMAGES
# ============================================================

augmented = df[
    df["filename"].str.contains("_aug_")
]

filenames = (
    augmented["filename"]
    .unique()
    .tolist()
)

random.seed(42)

random.shuffle(filenames)

# Visualize 30 examples
selected = filenames[:30]


print(
    f"Total augmented images: {len(filenames)}"
)

print(
    f"Visualizing: {len(selected)}"
)


# ============================================================
# DRAW BOXES
# ============================================================

for i, filename in enumerate(selected):

    image_path = IMAGE_DIR / filename

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"WARNING: Cannot read {filename}"
        )

        continue


    rows = augmented[
        augmented["filename"] == filename
    ]


    for _, row in rows.iterrows():

        xmin = int(row["xmin"])
        ymin = int(row["ymin"])
        xmax = int(row["xmax"])
        ymax = int(row["ymax"])

        label = str(
            row["class"]
        )


        # Bounding box
        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )


        # Label
        text = label

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.5

        thickness = 1


        text_size = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )[0]


        text_w, text_h = text_size


        # Background
        cv2.rectangle(
            image,
            (
                xmin,
                max(0, ymin - text_h - 6)
            ),
            (
                xmin + text_w + 4,
                ymin
            ),
            (0, 255, 0),
            -1
        )


        # Text
        cv2.putText(
            image,
            text,
            (
                xmin + 2,
                max(
                    text_h,
                    ymin - 3
                )
            ),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )


    output_name = (
        f"sample_{i+1:02d}_{filename}"
    )

    output_path = (
        OUTPUT_DIR / output_name
    )


    cv2.imwrite(
        str(output_path),
        image
    )


    print(
        f"[{i+1:02d}] {filename}"
    )


print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Visualizations saved to:\n"
    f"{OUTPUT_DIR}"
)
