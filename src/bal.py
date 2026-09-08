from pathlib import Path
from collections import Counter
import math
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "raw" / "yolo"
TRAIN_LABELS = OUT / "labels" / "train"
TRAIN_IMAGES = OUT / "images" / "train"
DATA_YAML = OUT / "data.yaml"
CHART_DIR = PROJECT_ROOT / "outputs"
CHART_DIR.mkdir(exist_ok=True)

MAX_OVERSAMPLE_FACTOR = 15   # cap how many times a minority image gets repeated


def main():
    names = yaml.safe_load(DATA_YAML.read_text())["names"]

    # count boxes per class, and which images contain which classes
    class_counts = Counter()
    image_classes = {}  # image stem -> set(class_ids)
    for label_file in TRAIN_LABELS.glob("*.txt"):
        ids = set()
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            cid = int(line.split()[0])
            class_counts[cid] += 1
            ids.add(cid)
        image_classes[label_file.stem] = ids

    if not class_counts:
        print("No labels found — run 01_prepare_yolo_data.py first.")
        return

    max_count = max(class_counts.values())
    print(f"{'class':<20}{'count':>8}{'imbalance ratio':>18}")
    for cid, cnt in sorted(class_counts.items(), key=lambda x: x[1]):
        ratio = max_count / cnt
        print(f"{names[cid]:<20}{cnt:>8}{ratio:>18.1f}")

    # per-class oversample factor = sqrt(max/count), capped
    class_factor = {
        cid: min(MAX_OVERSAMPLE_FACTOR, max(1, round(math.sqrt(max_count / cnt))))
        for cid, cnt in class_counts.items()
    }

    train_list = []
    for stem, ids in image_classes.items():
        img_path = TRAIN_IMAGES / f"{stem}.jpg"
        if not img_path.exists():
            img_path = TRAIN_IMAGES / f"{stem}.png"
        if not img_path.exists():
            continue
        repeat = max(class_factor[c] for c in ids) if ids else 1
        train_list.extend([img_path.as_posix()] * repeat)

    oversampled_txt = OUT / "train_oversampled.txt"
    oversampled_txt.write_text("\n".join(train_list))
    print(f"\nWrote {len(train_list)} entries (from {len(image_classes)} unique images) -> {oversampled_txt}")

    # plot
    fig, ax = plt.subplots(figsize=(12, 5))
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1])
    xs = range(len(sorted_items))
    ax.bar(xs, [n for _, n in sorted_items])
    ax.set_xticks(xs)
    ax.set_xticklabels([names[c] for c, _ in sorted_items], rotation=90, fontsize=6)
    ax.set_ylabel("box count")
    ax.set_title("Class distribution (train split)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "class_distribution.png", dpi=150)
    print(f"Saved chart -> {CHART_DIR / 'class_distribution.png'}")


if __name__ == "__main__":
    main()
