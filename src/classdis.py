from pathlib import Path
from collections import Counter

DATASET = Path("/home/admin1/Abishek-Intern/new_atr/data/balanced_700")

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


def count_classes(label_dir):

    counts = Counter()

    label_files = list(label_dir.glob("*.txt"))

    for label_file in label_files:

        with open(label_file, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                class_id = int(line.split()[0])

                counts[class_id] += 1

    return counts


for split in ["train", "validation", "test"]:

    label_dir = DATASET / "labels" / split

    print("\n" + "=" * 60)
    print(f"{split.upper()} DISTRIBUTION")
    print("=" * 60)

    if not label_dir.exists():
        print(f"Directory not found: {label_dir}")
        continue

    counts = count_classes(label_dir)

    total = sum(counts.values())

    print(f"\nTotal annotations: {total}")
    print(f"Label files: {len(list(label_dir.glob('*.txt')))}\n")

    print(
        f"{'ID':>3}  "
        f"{'Class':<10}  "
        f"{'Instances':>10}  "
        f"{'%':>8}"
    )

    print("-" * 40)

    for class_id, name in enumerate(CLASS_NAMES):

        count = counts.get(class_id, 0)

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"{class_id:>3}  "
            f"{name:<10}  "
            f"{count:>10}  "
            f"{percentage:>7.2f}%"
        )
