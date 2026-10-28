#!/usr/bin/env bash
set -u

PROJECT="/home/admin1/Abishek-Intern/new_atr"
cd "$PROJECT" || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$PROJECT/report_update_bundle_$STAMP"
mkdir -p "$OUT"

echo "Collecting project files..."

# 1. Main application and evaluation scripts
mkdir -p "$OUT/source_code"
for f in \
    4.py \
    evaluate_xai.py \
    check_anchor_alignment.py \
    merge_dataset.py \
    extract_tejas.py
do
    if [ -f "$PROJECT/$f" ]; then
        cp "$PROJECT/$f" "$OUT/source_code/"
    fi
done

# 2. Model training configuration and results (not model weights)
mkdir -p "$OUT/training"
MODEL_DIR="$PROJECT/models/yolov8s_832_79class_no_multiscale"

for f in args.yaml results.csv results.png confusion_matrix.png \
         confusion_matrix_normalized.png
do
    if [ -f "$MODEL_DIR/$f" ]; then
        cp "$MODEL_DIR/$f" "$OUT/training/"
    fi
done

# Include training result plots if available
if [ -d "$MODEL_DIR" ]; then
    find "$MODEL_DIR" -maxdepth 2 -type f \
        \( -name "PR_curve.png" -o -name "F1_curve.png" \
           -o -name "P_curve.png" -o -name "R_curve.png" \) \
        -exec cp -n {} "$OUT/training/" \; 2>/dev/null
fi

# 3. Dataset configuration
mkdir -p "$OUT/dataset"
for f in \
    "$PROJECT/data/yolo_79class_balanced_aug/data.yaml" \
    "$PROJECT/data/raw/yolo/data.yaml"
do
    if [ -f "$f" ]; then
        cp "$f" "$OUT/dataset/"
    fi
done

# 4. Latest XAI evaluation outputs
mkdir -p "$OUT/xai_evaluation"

LATEST_XAI=$(
    find "$PROJECT/runs" -maxdepth 1 -type d \
        -name 'xai_eval_50_*' -printf '%T@ %p\n' 2>/dev/null |
    sort -nr | head -n 1 | cut -d ' ' -f 2-
)

if [ -n "${LATEST_XAI:-}" ] && [ -d "$LATEST_XAI" ]; then
    echo "Latest XAI evaluation: $LATEST_XAI"
    cp "$LATEST_XAI"/xai_per_detection.csv "$OUT/xai_evaluation/" 2>/dev/null || true
    cp "$LATEST_XAI"/xai_method_summary.csv "$OUT/xai_evaluation/" 2>/dev/null || true
    printf '%s\n' "$LATEST_XAI" > "$OUT/xai_evaluation/source_folder.txt"
else
    echo "No XAI evaluation folder found."
fi

# 5. Anchor alignment diagnostic
if [ -f "$PROJECT/anchor_alignment_diagnostic.csv" ]; then
    cp "$PROJECT/anchor_alignment_diagnostic.csv" "$OUT/xai_evaluation/"
fi

# 6. Dataset image counts
python - "$PROJECT" "$OUT/dataset/dataset_counts.txt" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])

datasets = {
    "79-class dataset": root / "data/yolo_79class_balanced_aug/images",
    "Raw YOLO dataset": root / "data/raw/yolo/images",
}

with output.open("w") as f:
    for label, base in datasets.items():
        f.write(f"\n{label}\n")
        f.write(f"Path: {base}\n")
        if not base.is_dir():
            f.write("Image directory not found.\n")
            continue

        total = 0
        for split in ("train", "validation", "val", "test"):
            folder = base / split
            if not folder.is_dir():
                continue
            count = sum(
                1 for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in
                {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
            f.write(f"{split}: {count}\n")
            total += count
        f.write(f"Total across listed splits: {total}\n")
PY

# 7. Git status and repository information
mkdir -p "$OUT/repository_info"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git status --short > "$OUT/repository_info/git_status.txt" 2>&1
    git log -1 --format=fuller > "$OUT/repository_info/latest_commit.txt" 2>&1
    git remote -v > "$OUT/repository_info/git_remotes.txt" 2>&1
else
    echo "Git repository information unavailable." \
        > "$OUT/repository_info/git_status.txt"
fi

# 8. Project summary for report preparation
cat > "$OUT/PROJECT_SUMMARY.txt" <<'EOF'
AIRCRAFT DETECTION AND EXPLAINABLE AI PROJECT

Main application: 4.py

Model:
YOLOv8s, 79 aircraft classes
Model directory:
models/yolov8s_832_79class_no_multiscale/

Training configuration previously reported:
Epochs: 150
Image size: 832
Batch size: 48
Multiscale training: disabled

Latest reported test metrics:
Precision: 0.873
Recall: 0.718
mAP@0.5: 0.817
mAP@0.5:0.95: 0.708

Latest reported XAI evaluation:
Sample: 50 selected test images
Detections: 84
Occlusion mean combined score: 0.5633
SmoothGrad mean combined score: 0.5228
EigenCAM mean combined score: 0.0515
XGradCAM mean combined score: 0.0370

Selection frequency:
Occlusion: 71.4%
SmoothGrad: 28.6%
EigenCAM: 0%
XGradCAM: 0%

IMPORTANT:
XAI results are provisional pending validation of internal prediction
matching and faithfulness calculations. Do not present these results as
fully validated without reviewing the included scripts and CSV files.

The diagnostic anchor-alignment run reported 88 detections, mean highest
box IoU 0.9798, and mean selected-anchor IoU 0.9624. Some individual
internal class scores differed substantially from YOLO detection confidence.
EOF

# 9. Optional existing report
echo
echo "If your existing report is a PDF, DOCX, or other file, enter its full path."
echo "If you do not have it available now, just press Enter."
read -r -p "Existing report path: " REPORT_PATH

if [ -n "$REPORT_PATH" ] && [ -f "$REPORT_PATH" ]; then
    mkdir -p "$OUT/existing_report"
    cp "$REPORT_PATH" "$OUT/existing_report/"
    echo "Existing report included."
else
    echo "No existing report included. Upload it separately if needed."
fi

# 10. Zip the bundle
ARCHIVE="$PROJECT/report_update_bundle_$STAMP.zip"

python - "$OUT" "$ARCHIVE" <<'PY'
import sys
import zipfile
from pathlib import Path

folder = Path(sys.argv[1])
archive = Path(sys.argv[2])

with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for p in folder.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(folder))
print(f"\nBundle created: {archive}")
print(f"Files collected: {sum(1 for p in folder.rglob('*') if p.is_file())}")
PY

echo
echo "============================================"
echo "REPORT COLLECTION COMPLETE"
echo "============================================"
echo "Archive: $ARCHIVE"
echo
echo "Upload this ZIP file to ChatGPT."
echo "Model weights and the image dataset were not copied."
