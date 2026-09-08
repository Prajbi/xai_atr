from pathlib import Path
from ultralytics import YOLO

# ============================================================
# MODELS
# ============================================================

MODELS = {
    "YOLOv8s 832 350+": "/home/admin1/Abishek-Intern/new_atr/runs/yolov8s_832_350plus/weights/best.pt",

    "YOLOv8s 832 balanced700": "/home/admin1/Abishek-Intern/new_atr/runs/yolov8s_832_balanced700/weights/best.pt",

    "YOLOv8s 640 350+": "/home/admin1/Abishek-Intern/new_atr/runs/yolov8s_350plus/weights/best.pt",
}

# ============================================================
# DATASET
# ============================================================

DATA_YAML = "/home/admin1/Abishek-Intern/new_atr/data/balanced_700/data.yaml"

# ============================================================
# SETTINGS
# ============================================================

IMGSZ = 832
BATCH = 64
DEVICE = 0
WORKERS = 8


# ============================================================
# EVALUATE
# ============================================================

results = {}

for name, model_path in MODELS.items():

    print("\n")
    print("=" * 80)
    print(name)
    print("=" * 80)

    if not Path(model_path).exists():
        print(f"ERROR: Model not found:")
        print(model_path)
        continue

    print(f"\nModel: {model_path}")

    model = YOLO(model_path)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        verbose=True,
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    results[name] = {
        "Precision": precision,
        "Recall": recall,
        "mAP50": map50,
        "mAP50-95": map50_95,
    }

    print("\nRESULT:")
    print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"mAP50     : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"mAP50-95  : {map50_95:.4f} ({map50_95 * 100:.2f}%)")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n")
print("=" * 100)
print("                         FINAL MODEL COMPARISON")
print("=" * 100)

print(
    f"{'Model':<30}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'mAP50':>12}"
    f"{'mAP50-95':>14}"
)

print("-" * 100)

for name, r in results.items():

    print(
        f"{name:<30}"
        f"{r['Precision'] * 100:>11.2f}%"
        f"{r['Recall'] * 100:>11.2f}%"
        f"{r['mAP50'] * 100:>11.2f}%"
        f"{r['mAP50-95'] * 100:>13.2f}%"
    )

print("=" * 100)
