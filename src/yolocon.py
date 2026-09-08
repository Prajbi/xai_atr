import os
import shutil
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # folder containing src/, data/, outputs/, runs/
DATA_ROOT = PROJECT_ROOT / "data/raw"
IMG_DIR = DATA_ROOT / "dataset"
CSV_PATH = DATA_ROOT / "labels_with_split.csv"        # put the CSV directly here, not in a subfolder
OUT = DATA_ROOT / "yolo"

COL_ALIASES = {
    "filename": ["filename", "image", "image_id", "file", "id"],
    "width":    ["width", "img_width", "w"],
    "height":   ["height", "img_height", "h"],
    "class":    ["class", "label", "class_name", "aircraft"],
    "xmin":     ["xmin", "x_min", "xmin_pixel"],
    "ymin":     ["ymin", "y_min", "ymin_pixel"],
    "xmax":     ["xmax", "x_max", "xmax_pixel"],
    "ymax":     ["ymax", "y_max", "ymax_pixel"],
    "split":    ["split", "set", "subset"],
}


def resolve_columns(df):
    cols_lower = {c.lower(): c for c in df.columns}
    resolved = {}
    for canon, aliases in COL_ALIASES.items():
        for a in aliases:
            if a in cols_lower:
                resolved[canon] = cols_lower[a]
                break
        else:
            if canon != "split":
                raise ValueError(f"Could not find a column for '{canon}'. Columns present: {list(df.columns)}")
    return resolved


def main():
    df = pd.read_csv(CSV_PATH)
    col = resolve_columns(df)

    has_split = "split" in col
    if not has_split:
        # deterministic 80/10/10 split by filename hash
        import hashlib
        def bucket(name):
            h = int(hashlib.md5(str(name).encode()).hexdigest(), 16) % 100
            if h < 80: return "train"
            if h < 90: return "val"
            return "test"
        df["split"] = df[col["filename"]].apply(bucket)
        col["split"] = "split"

    classes = sorted(df[col["class"]].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"Found {len(classes)} classes.")

    for split in ["train", "val", "test"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(col["filename"])
    n_written, n_missing_img = 0, 0

    for fname, group in grouped:
        split = str(group.iloc[0][col["split"]]).lower()
        split = {"validation": "val", "valid": "val"}.get(split, split)
        split = split if split in ("train", "val", "test") else "train"

        stem = Path(str(fname)).stem
        src_img = IMG_DIR / f"{stem}.jpg"
        if not src_img.exists():
            src_img = IMG_DIR / f"{stem}.png"
        if not src_img.exists():
            n_missing_img += 1
            continue

        dst_img = OUT / "images" / split / src_img.name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        lines = []
        for _, row in group.iterrows():
            w, h = float(row[col["width"]]), float(row[col["height"]])
            xmin, ymin = float(row[col["xmin"]]), float(row[col["ymin"]])
            xmax, ymax = float(row[col["xmax"]]), float(row[col["ymax"]])
            cx = ((xmin + xmax) / 2) / w
            cy = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            cid = class_to_id[row[col["class"]]]
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = OUT / "labels" / split / f"{stem}.txt"
        label_path.write_text("\n".join(lines))
        n_written += 1

    yaml_text = "path: {}\n".format(OUT.as_posix())
    yaml_text += "train: images/train\nval: images/val\ntest: images/test\n"
    yaml_text += f"nc: {len(classes)}\n"
    yaml_text += "names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(classes)) + "\n"
    (OUT / "data.yaml").write_text(yaml_text)

    print(f"Wrote {n_written} label files, {n_missing_img} images missing on disk.")
    print(f"data.yaml -> {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
