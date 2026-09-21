from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")

DATA_YAML = PROJECT_ROOT / "data/yolo_aircraft_89/data.yaml"

BEST_WEIGHTS = (
    PROJECT_ROOT
    / "new/models/yolo_aircraft.pt"
)


print("\n========================================")
print("       YOLOv8s TEST EVALUATION")
print("========================================")

print(f"\nModel:")
print(BEST_WEIGHTS)

print(f"\nDataset:")
print(DATA_YAML)


if not BEST_WEIGHTS.exists():
    raise FileNotFoundError(
        f"best.pt not found:\n{BEST_WEIGHTS}"
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"data.yaml not found:\n{DATA_YAML}"
    )


# Load trained best model
model = YOLO(BEST_WEIGHTS.as_posix())


# Evaluate ONLY the untouched TEST set
metrics = model.val(
    data=DATA_YAML.as_posix(),
    split="val",

    imgsz=640,
    batch=64,

    device=0,
    workers=8,

    verbose=True,
)


print("\n========================================")
print("          FINAL TEST METRICS")
print("========================================")

print(f"\nmAP50-95 : {metrics.box.map:.4f}")
print(f"mAP50    : {metrics.box.map50:.4f}")

if hasattr(metrics.box, "map75"):
    print(f"mAP75    : {metrics.box.map75:.4f}")

print("\nPer-class mAP50-95:")
print(metrics.box.maps)

print("\n========================================")
print("             TEST DONE")
print("========================================")
