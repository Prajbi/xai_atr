import pandas as pd, shutil, argparse, yaml
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/raw/labels_with_split.csv")
ap.add_argument("--images", default="data/raw/dataset")
ap.add_argument("--out", default="data/yolo")
args = ap.parse_args()

df = pd.read_csv(args.csv)
classes = sorted(df["class"].unique())
cls2idx = {c: i for i, c in enumerate(classes)}

images_dir = Path(args.images)
out = Path(args.out)
for split in df["split"].unique():
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

print("Indexing image files...")
img_index = {}
for p in images_dir.iterdir():
    if p.suffix.lower() != ".csv":
        img_index.setdefault(p.stem, p)
print(f"Indexed {len(img_index)} image files")

skipped = 0
copied = 0
for fname, group in df.groupby("filename"):
    split = group["split"].iloc[0]
    img_src = img_index.get(fname)
    if img_src is None:
        skipped += 1
        continue
    shutil.copy2(img_src, out / "images" / split / img_src.name)
    lines = []
    for _, r in group.iterrows():
        cx = ((r.xmin + r.xmax) / 2) / r.width
        cy = ((r.ymin + r.ymax) / 2) / r.height
        bw = (r.xmax - r.xmin) / r.width
        bh = (r.ymax - r.ymin) / r.height
        lines.append(f"{cls2idx[r['class']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (out / "labels" / split / (img_src.stem + ".txt")).write_text("\n".join(lines))
    copied += 1
    if copied % 2000 == 0:
        print(f"...{copied} copied")

yaml.safe_dump({"path": str(out.resolve()), "train": "images/train",
                 "val": "images/validation", "nc": len(classes), "names": classes},
                open("configs/data.yaml", "w"))
print(f"{len(classes)} classes, splits: {df['split'].unique().tolist()}")
print(f"copied {copied} images, skipped {skipped} (not found in {images_dir})")
