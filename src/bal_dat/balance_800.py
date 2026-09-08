from pathlib import Path
import shutil
import random
import cv2
import numpy as np
import yaml

# ============================================================
# CONFIG
# ============================================================

SOURCE = Path("/home/admin1/Abishek-Intern/new_atr/data/filtered_100")
OUTPUT = Path("/home/admin1/Abishek-Intern/new_atr/data/balanced_800")

TARGET = 800
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# ============================================================
# HELPERS
# ============================================================

def find_image(image_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_yolo_label(label_path):
    labels = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, xc, yc, w, h = parts

            labels.append([
                int(cls),
                float(xc),
                float(yc),
                float(w),
                float(h)
            ])

    return labels


def save_yolo_label(path, labels):
    with open(path, "w") as f:
        for cls, xc, yc, w, h in labels:
            f.write(
                f"{cls} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


# ============================================================
# AUGMENTATION
# ============================================================

def augment_image(image, labels):

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # 1. Horizontal flip
    # --------------------------------------------------------

    if random.random() < 0.5:

        image = cv2.flip(image, 1)

        for label in labels:
            label[1] = 1.0 - label[1]


    # --------------------------------------------------------
    # 2. Small rotation
    # --------------------------------------------------------

    angle = random.uniform(-8, 8)

    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )

    # Transform bounding boxes
    for label in labels:

        cls, xc, yc, bw, bh = label

        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h

        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])

        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])

        transformed = corners_h @ matrix.T

        nx1 = transformed[:, 0].min()
        ny1 = transformed[:, 1].min()
        nx2 = transformed[:, 0].max()
        ny2 = transformed[:, 1].max()

        nx1 = max(0, min(w, nx1))
        ny1 = max(0, min(h, ny1))
        nx2 = max(0, min(w, nx2))
        ny2 = max(0, min(h, ny2))

        nw = nx2 - nx1
        nh = ny2 - ny1

        if nw <= 1 or nh <= 1:
            continue

        nxc = (nx1 + nx2) / 2 / w
        nyc = (ny1 + ny2) / 2 / h

        label[1] = max(0, min(1, nxc))
        label[2] = max(0, min(1, nyc))
        label[3] = max(0, min(1, nw / w))
        label[4] = max(0, min(1, nh / h))


    # --------------------------------------------------------
    # 3. Brightness / contrast
    # --------------------------------------------------------

    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-20, 20)

    image = cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta
    )


    # --------------------------------------------------------
    # 4. Mild blur
    # --------------------------------------------------------

    if random.random() < 0.15:
        image = cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )


    # --------------------------------------------------------
    # 5. Mild noise
    # --------------------------------------------------------

    if random.random() < 0.10:

        noise = np.random.normal(
            0,
            5,
            image.shape
        ).astype(np.int16)

        image = np.clip(
            image.astype(np.int16) + noise,
            0,
            255
        ).astype(np.uint8)


    return image, labels


# ============================================================
# CREATE DIRECTORIES
# ============================================================

print("=" * 70)
print("CREATING BALANCED 800 DATASET")
print("=" * 70)

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
# READ YAML
# ============================================================

with open(SOURCE / "data.yaml") as f:
    data = yaml.safe_load(f)

names = data["names"]

print(f"Classes: {len(names)}")


# ============================================================
# COPY VALIDATION + TEST UNCHANGED
# ============================================================

print()
print("=" * 70)
print("COPYING VALIDATION / TEST WITHOUT MODIFICATION")
print("=" * 70)

for split in ["validation", "test"]:

    source_images = SOURCE / "images" / split
    source_labels = SOURCE / "labels" / split

    output_images = OUTPUT / "images" / split
    output_labels = OUTPUT / "labels" / split

    for image in source_images.iterdir():

        if not image.is_file():
            continue

        shutil.copy2(
            image,
            output_images / image.name
        )

    for label in source_labels.glob("*.txt"):

        shutil.copy2(
            label,
            output_labels / label.name
        )

    print(f"{split}: copied")


# ============================================================
# TRAINING DATA
# ============================================================

TRAIN_IMAGES = SOURCE / "images" / "train"
TRAIN_LABELS = SOURCE / "labels" / "train"

OUTPUT_TRAIN_IMAGES = OUTPUT / "images" / "train"
OUTPUT_TRAIN_LABELS = OUTPUT / "labels" / "train"


# ============================================================
# COPY ORIGINAL TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("COPYING ORIGINAL TRAINING DATA")
print("=" * 70)

for image in TRAIN_IMAGES.iterdir():

    if not image.is_file():
        continue

    shutil.copy2(
        image,
        OUTPUT_TRAIN_IMAGES / image.name
    )

for label in TRAIN_LABELS.glob("*.txt"):

    shutil.copy2(
        label,
        OUTPUT_TRAIN_LABELS / label.name
    )


# ============================================================
# BUILD CLASS → IMAGES INDEX
# ============================================================

class_images = {
    class_id: []
    for class_id in range(len(names))
}

for label_path in TRAIN_LABELS.glob("*.txt"):

    labels = load_yolo_label(label_path)

    classes_in_image = set(
        label[0]
        for label in labels
    )

    for class_id in classes_in_image:

        if class_id in class_images:
            class_images[class_id].append(
                label_path.stem
            )


# ============================================================
# CURRENT COUNTS
# ============================================================

print()
print("=" * 70)
print("CURRENT TRAINING DISTRIBUTION")
print("=" * 70)

for class_id in range(len(names)):

    count = len(class_images[class_id])

    print(
        f"{class_id:2d} "
        f"{names[class_id]:10s} "
        f"{count:5d}"
    )


# ============================================================
# TARGETED AUGMENTATION
# ============================================================

print()
print("=" * 70)
print("STARTING TARGETED AUGMENTATION")
print("=" * 70)

augmentation_counts = {
    class_id: 0
    for class_id in range(len(names))
}


for class_id in range(len(names)):

    current_count = len(class_images[class_id])

    if current_count >= TARGET:

        print(
            f"{names[class_id]:10s}: "
            f"{current_count} → {current_count} "
            f"(no augmentation)"
        )

        continue

    needed = TARGET - current_count

    print(
        f"{names[class_id]:10s}: "
        f"{current_count} → {TARGET} "
        f"(creating {needed})"
    )

    candidates = class_images[class_id].copy()

    random.shuffle(candidates)

    for i in range(needed):

        stem = candidates[i % len(candidates)]

        image_path = find_image(
            TRAIN_IMAGES,
            stem
        )

        label_path = TRAIN_LABELS / f"{stem}.txt"

        if image_path is None:
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        labels = load_yolo_label(
            label_path
        )

        # Make a deep copy
        labels = [
            label.copy()
            for label in labels
        ]

        augmented_image, augmented_labels = augment_image(
            image,
            labels
        )

        new_stem = (
            f"{stem}_aug_{class_id}_{i:04d}"
        )

        output_image = (
            OUTPUT_TRAIN_IMAGES
            / f"{new_stem}{image_path.suffix}"
        )

        output_label = (
            OUTPUT_TRAIN_LABELS
            / f"{new_stem}.txt"
        )

        cv2.imwrite(
            str(output_image),
            augmented_image
        )

        save_yolo_label(
            output_label,
            augmented_labels
        )

        augmentation_counts[class_id] += 1


# ============================================================
# CREATE YAML
# ============================================================

yaml_output = OUTPUT / "data.yaml"

yaml_data = {
    "path": str(OUTPUT),
    "train": "images/train",
    "val": "images/validation",
    "test": "images/test",
    "nc": len(names),
    "names": names
}

with open(yaml_output, "w") as f:
    yaml.safe_dump(
        yaml_data,
        f,
        sort_keys=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET")
print("=" * 70)

for class_id in range(len(names)):

    original = len(class_images[class_id])
    augmented = augmentation_counts[class_id]

    print(
        f"{names[class_id]:10s} "
        f"{original:4d} + "
        f"{augmented:4d} = "
        f"{original + augmented:4d}"
    )


train_images = list(
    OUTPUT_TRAIN_IMAGES.iterdir()
)

val_images = list(
    (OUTPUT / "images" / "validation").iterdir()
)

test_images = list(
    (OUTPUT / "images" / "test").iterdir()
)

print()
print(f"Train images      : {len(train_images):,}")
print(f"Validation images : {len(val_images):,}")
print(f"Test images       : {len(test_images):,}")
print(f"YAML              : {yaml_output}")

print()
print("=" * 70)
print("BALANCED DATASET CREATED")
print("=" * 70)
