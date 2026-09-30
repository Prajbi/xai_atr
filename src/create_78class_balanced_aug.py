from pathlib import Path
from collections import Counter, defaultdict
import random
import shutil
import cv2
import yaml
import albumentations as A


# ============================================================
# CONFIG
# ============================================================

SOURCE = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/yolo_78class"
)

OUTPUT = Path(
    "/home/admin1/Abishek-Intern/new_atr/data/yolo_78class_balanced_aug"
)

SEED = 42
random.seed(SEED)

# ============================================================
# BASELINE PERFORMANCE
# ============================================================

RECALL = {
    "A10": .566, "A400M": .743, "AG600": .766, "AH64": .804,
    "AV8B": .791, "An124": .815, "An22": .812, "An225": .472,
    "An72": .489, "B1": .793, "B2": .803, "B52": .711,
    "Be200": .864, "C1": .780, "C130": .675, "C17": .655,
    "C2": .817, "C390": .741, "C5": .786, "CH47": .787,
    "CL415": .795, "E2": .766, "E7": .878, "EF2000": .619,
    "EMB314": .690, "F117": .716, "F14": .516, "F15": .682,
    "F16": .546, "F18": .738, "F2": .396, "F22": .629,
    "F35": .660, "F4": .582, "H6": .818, "Il76": .426,
    "J10": .553, "J20": .663, "J35": .423, "JAS39": .617,
    "JF17": .690, "JH7": .680, "KC135": .729, "KF21": .607,
    "KJ600": .625, "Ka27": .543, "Ka52": .720, "MQ9": .783,
    "Mi24": .727, "Mi26": .675, "Mi28": .756, "Mi8": .587,
    "Mig29": .299, "Mig31": .783, "Mirage2000": .607, "P3": .741,
    "RQ4": .717, "Rafale": .637, "SR71": .787, "Su24": .584,
    "Su25": .642, "Su34": .667, "Su47": .592, "Su57": .691,
    "TB2": .449, "Tornado": .646, "Tu160": .889, "Tu22M": .608,
    "Tu95": .814, "U2": .893, "UH60": .476, "US2": .926,
    "V22": .685, "Vulcan": .694, "WZ7": .503, "XB70": .844,
    "Y20": .579, "YF23": .568,
}

MAP95 = {
    "A10": .581, "A400M": .725, "AG600": .756, "AH64": .662,
    "AV8B": .767, "An124": .510, "An22": .461, "An225": .367,
    "An72": .450, "B1": .698, "B2": .680, "B52": .665,
    "Be200": .826, "C1": .695, "C130": .625, "C17": .623,
    "C2": .859, "C390": .681, "C5": .637, "CH47": .723,
    "CL415": .751, "E2": .638, "E7": .747, "EF2000": .670,
    "EMB314": .482, "F117": .670, "F14": .507, "F15": .636,
    "F16": .503, "F18": .646, "F2": .202, "F22": .626,
    "F35": .598, "F4": .588, "H6": .722, "Il76": .450,
    "J10": .538, "J20": .684, "J35": .310, "JAS39": .606,
    "JF17": .558, "JH7": .738, "KC135": .609, "KF21": .419,
    "KJ600": .601, "Ka27": .541, "Ka52": .724, "MQ9": .685,
    "Mi24": .732, "Mi26": .496, "Mi28": .563, "Mi8": .441,
    "Mig29": .334, "Mig31": .717, "Mirage2000": .585, "P3": .690,
    "RQ4": .519, "Rafale": .631, "SR71": .720, "Su24": .609,
    "Su25": .589, "Su34": .610, "Su47": .532, "Su57": .642,
    "TB2": .380, "Tornado": .701, "Tu160": .806, "Tu22M": .572,
    "Tu95": .717, "U2": .732, "UH60": .393, "US2": .861,
    "V22": .584, "Vulcan": .683, "WZ7": .371, "XB70": .704,
    "Y20": .550, "YF23": .492,
}


# ============================================================
# AUGMENTATION
# ============================================================

MODERATE = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.92, 1.08),
            translate_percent=(-0.04, 0.04),
            rotate=(-7, 7),
            shear=(-2, 2),
            p=0.65,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.45,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.30,
        ),
        A.GaussNoise(p=0.10),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        filter_invalid_bboxes=True,
        min_visibility=0.20,
    ),
)

STRONG = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.88, 1.12),
            translate_percent=(-0.06, 0.06),
            rotate=(-9, 9),
            shear=(-3, 3),
            p=0.70,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.20,
            p=0.50,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=18,
            val_shift_limit=12,
            p=0.35,
        ),
        A.OneOf(
            [
                A.GaussNoise(p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
            ],
            p=0.15,
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        filter_invalid_bboxes=True,
        min_visibility=0.20,
    ),
)


# ============================================================
# HELPERS
# ============================================================

def read_labels(path):
    boxes = []
    labels = []

    with open(path) as f:
        for line in f:
            p = line.strip().split()

            if len(p) != 5:
                continue

            cls = int(p[0])

            x = float(p[1])
            y = float(p[2])
            w = float(p[3])
            h = float(p[4])

            # Fix tiny floating point excursions.
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            # Make sure center-width representation is valid.
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            x1 = min(max(x1, 0.0), 1.0)
            y1 = min(max(y1, 0.0), 1.0)
            x2 = min(max(x2, 0.0), 1.0)
            y2 = min(max(y2, 0.0), 1.0)

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2

            boxes.append([x, y, w, h])
            labels.append(cls)

    return boxes, labels


def write_labels(path, boxes, labels):
    with open(path, "w") as f:
        for box, cls in zip(boxes, labels):
            x, y, w, h = box

            x = min(max(float(x), 0.0), 1.0)
            y = min(max(float(y), 0.0), 1.0)
            w = min(max(float(w), 0.0), 1.0)
            h = min(max(float(h), 0.0), 1.0)

            if w <= 0 or h <= 0:
                continue

            f.write(
                f"{cls} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


def priority(name):
    """
    Higher = more attention.
    """

    r = RECALL[name]
    m = MAP95[name]

    return (
        0.55 * (1 - r)
        + 0.45 * (1 - m)
    )


def target_for(name, current):
    """
    Three-level strategy.

    Very weak:
        target 400

    Moderate:
        target 350

    Better-performing minority:
        target 300
    """

    r = RECALL[name]
    m = MAP95[name]

    if r < 0.50 or m < 0.40:
        return 400

    if r < 0.65 or m < 0.55:
        return 350

    return 300


# ============================================================
# LOAD YAML
# ============================================================

with open(SOURCE / "data.yaml") as f:
    data = yaml.safe_load(f)

names = data["names"]

if isinstance(names, dict):
    names = [names[i] for i in range(len(names))]

nc = len(names)


# ============================================================
# RESET OUTPUT
# ============================================================

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split in ["train", "validation", "test"]:
    (OUTPUT / "images" / split).mkdir(parents=True)
    (OUTPUT / "labels" / split).mkdir(parents=True)


# ============================================================
# COPY VALIDATION + TEST
# ============================================================

for split in ["validation", "test"]:

    src_i = SOURCE / "images" / split
    src_l = SOURCE / "labels" / split

    dst_i = OUTPUT / "images" / split
    dst_l = OUTPUT / "labels" / split

    for img in src_i.iterdir():

        if not img.is_file():
            continue

        label = src_l / f"{img.stem}.txt"

        shutil.copy2(img, dst_i / img.name)

        if label.exists():
            shutil.copy2(
                label,
                dst_l / label.name
            )

    print(
        f"{split}: "
        f"{len(list(dst_i.iterdir()))} images"
    )


# ============================================================
# LOAD TRAINING INFORMATION
# ============================================================

train_i = SOURCE / "images" / "train"
train_l = SOURCE / "labels" / "train"

class_images = defaultdict(list)
class_counts = Counter()

image_infos = []

for img in train_i.iterdir():

    if not img.is_file():
        continue

    label = train_l / f"{img.stem}.txt"

    if not label.exists():
        continue

    boxes, labels = read_labels(label)

    if not labels:
        continue

    classes = set(labels)

    info = {
        "image": img,
        "label": label,
        "boxes": boxes,
        "labels": labels,
        "classes": classes,
    }

    image_infos.append(info)

    for cls in classes:
        class_images[cls].append(info)
        class_counts[cls] += 1


# ============================================================
# COPY ORIGINAL TRAINING DATA
# ============================================================

for info in image_infos:

    shutil.copy2(
        info["image"],
        OUTPUT / "images" / "train" / info["image"].name
    )

    shutil.copy2(
        info["label"],
        OUTPUT / "labels" / "train" / info["label"].name
    )


# ============================================================
# TARGETS
# ============================================================

targets = {}

for cls_id, name in enumerate(names):

    current = class_counts[cls_id]

    if current < 300:
        targets[cls_id] = target_for(
            name,
            current
        )
    else:
        targets[cls_id] = current


print("\n" + "=" * 90)
print("TARGETS")
print("=" * 90)

for cls_id, name in enumerate(names):

    print(
        f"{name:<15} "
        f"current={class_counts[cls_id]:>4} "
        f"target={targets[cls_id]:>4} "
        f"recall={RECALL[name]:.3f} "
        f"mAP95={MAP95[name]:.3f}"
    )


# ============================================================
# OVERSAMPLING
# ============================================================

current = Counter(class_counts)

generated = 0
attempts = 0

print("\nGenerating augmented samples...")


while True:

    deficits = {
        cls_id: targets[cls_id] - current[cls_id]
        for cls_id in targets
        if current[cls_id] < targets[cls_id]
    }

    if not deficits:
        break

    attempts += 1

    if attempts > 100000:
        print("Safety stop.")
        break

    # Select class with largest weighted deficit.
    selected_cls = max(
        deficits,
        key=lambda c:
        deficits[c] *
        (1 + priority(names[c]))
    )

    candidates = class_images[selected_cls]

    if not candidates:
        current[selected_cls] = targets[selected_cls]
        continue

    # Prefer candidates containing several minority classes.
    best = []

    for info in candidates:

        score = 1.0

        for cls in info["classes"]:

            if cls in deficits:
                score += deficits[cls]

        best.append((info, score))

    total = sum(x[1] for x in best)

    r = random.uniform(0, total)

    acc = 0
    selected = None

    for info, weight in best:

        acc += weight

        if acc >= r:
            selected = info
            break

    if selected is None:
        selected = random.choice(candidates)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        str(selected["image"])
    )

    if image is None:
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    boxes = selected["boxes"]
    labels = selected["labels"]

    # --------------------------------------------------------
    # Select augmentation strength
    # --------------------------------------------------------

    relevant = [
        cls
        for cls in selected["classes"]
        if cls in deficits
    ]

    if not relevant:
        continue

    strongest = max(
        relevant,
        key=lambda c: priority(names[c])
    )

    if (
        RECALL[names[strongest]] < 0.50
        or MAP95[names[strongest]] < 0.40
    ):
        augmenter = STRONG
    else:
        augmenter = MODERATE

    # --------------------------------------------------------
    # Augment
    # --------------------------------------------------------

    try:

        result = augmenter(
            image=image,
            bboxes=boxes,
            class_labels=labels
        )

    except Exception:
        continue

    new_boxes = result["bboxes"]
    new_labels = result["class_labels"]

    if not new_boxes:
        continue

    # --------------------------------------------------------
    # Unique filename
    # --------------------------------------------------------

    new_id = f"aug_{generated:07d}"

    image_path = (
        OUTPUT /
        "images" /
        "train" /
        f"{new_id}.jpg"
    )

    label_path = (
        OUTPUT /
        "labels" /
        "train" /
        f"{new_id}.txt"
    )

    new_image = cv2.cvtColor(
        result["image"],
        cv2.COLOR_RGB2BGR
    )

    ok = cv2.imwrite(
        str(image_path),
        new_image
    )

    if not ok:
        continue

    write_labels(
        label_path,
        new_boxes,
        new_labels
    )

    # --------------------------------------------------------
    # Update all classes present
    # --------------------------------------------------------

    for cls in set(new_labels):
        current[cls] += 1

    generated += 1

    if generated % 250 == 0:

        remaining = sum(
            max(
                0,
                targets[c] - current[c]
            )
            for c in targets
        )

        print(
            f"Generated {generated:>5} | "
            f"remaining deficit {remaining:>5}"
        )


# ============================================================
# DATA YAML
# ============================================================

out_yaml = {
    "path": str(OUTPUT),
    "train": "images/train",
    "val": "images/validation",
    "test": "images/test",
    "nc": nc,
    "names": names,
}

with open(OUTPUT / "data.yaml", "w") as f:
    yaml.safe_dump(
        out_yaml,
        f,
        sort_keys=False
    )


# ============================================================
# FINAL COUNTS
# ============================================================

final_counts = Counter()
final_boxes = Counter()

final_label_dir = OUTPUT / "labels" / "train"

for label in final_label_dir.glob("*.txt"):

    boxes, labels = read_labels(label)

    for cls in set(labels):
        final_counts[cls] += 1

    for cls in labels:
        final_boxes[cls] += 1


final_images = len(
    list(
        (OUTPUT / "images" / "train").iterdir()
    )
)

total_boxes = sum(final_boxes.values())


print("\n" + "=" * 90)
print("FINAL DATASET")
print("=" * 90)

print(f"Training images : {final_images}")
print(f"Training boxes  : {total_boxes}")
print(f"Generated       : {generated}")

print("\n")
print(
    f"{'CLASS':<15}"
    f"{'BEFORE':>10}"
    f"{'TARGET':>10}"
    f"{'AFTER':>10}"
    f"{'BOXES':>10}"
)

print("-" * 60)

for cls_id, name in enumerate(names):

    print(
        f"{name:<15}"
        f"{class_counts[cls_id]:>10}"
        f"{targets[cls_id]:>10}"
        f"{final_counts[cls_id]:>10}"
        f"{final_boxes[cls_id]:>10}"
    )

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
print(f"Dataset: {OUTPUT}")
