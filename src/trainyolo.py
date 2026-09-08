from pathlib import Path
import yaml
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")

OUT = PROJECT_ROOT / "data" / "raw" / "yolo"

DATA_YAML = OUT / "data.yaml"
OVERSAMPLED_TRAIN_LIST = OUT / "train_oversampled.txt"
TRAIN_YAML = OUT / "data_oversampled.yaml"

RUNS_DIR = PROJECT_ROOT / "runs"
RUN_NAME = "yolov8s_testnew"

# ============================================================
# LOCAL YOLOv8s WEIGHTS
# ============================================================

WEIGHTS = PROJECT_ROOT / "yolov8s.pt"


# ============================================================
# VERIFY LOCAL MODEL
# ============================================================

print("\n========================================")
print("       CHECKING LOCAL YOLOv8s")
print("========================================")

print(f"\nModel path:")
print(WEIGHTS)

print(f"\nExists: {WEIGHTS.exists()}")

if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"\nYOLOv8s weights not found:\n{WEIGHTS}\n\n"
        "Run:\n"
        "ls -lh /home/admin1/Abishek-Intern/new_atr/*.pt"
    )

file_size_mb = WEIGHTS.stat().st_size / (1024 * 1024)

print(f"Size: {file_size_mb:.2f} MB")

if WEIGHTS.stat().st_size == 0:
    raise RuntimeError(
        f"YOLOv8s weights file is empty:\n{WEIGHTS}"
    )

print("\nLocal YOLOv8s file found.")


# ============================================================
# VERIFY DATA YAML
# ============================================================

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"\ndata.yaml not found:\n{DATA_YAML}"
    )


# ============================================================
# LOAD DATA YAML
# ============================================================

base = yaml.safe_load(
    DATA_YAML.read_text()
)

if not isinstance(base, dict):
    raise ValueError(
        f"Invalid data.yaml:\n{DATA_YAML}"
    )


# ============================================================
# USE OVERSAMPLED TRAINING LIST
# ============================================================

if OVERSAMPLED_TRAIN_LIST.exists():

    base["train"] = OVERSAMPLED_TRAIN_LIST.as_posix()

    print("\n========================================")
    print("       OVERSAMPLED TRAINING")
    print("========================================")

    print("\nUsing:")
    print(OVERSAMPLED_TRAIN_LIST)

else:

    print("\nWARNING:")
    print("train_oversampled.txt was not found.")
    print("Using original training dataset.")


# ============================================================
# WRITE TRAINING YAML
# ============================================================

TRAIN_YAML.write_text(
    yaml.safe_dump(
        base,
        sort_keys=False
    )
)

print("\nTraining YAML created:")
print(TRAIN_YAML)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("          YOLOv8s TRAINING")
    print("========================================")

    print(f"\nWeights : {WEIGHTS}")
    print(f"Data    : {TRAIN_YAML}")
    print(f"Epochs  : 50")
    print(f"Image   : 832")
    print(f"Batch   : 64")
    print(f"Device  : 0")


    # ========================================================
    # LOAD LOCAL MODEL
    # ========================================================

    print("\nLoading LOCAL YOLOv8s...")

    model = YOLO(
        WEIGHTS.as_posix()
    )

    print("Local YOLOv8s loaded successfully.")


    # ========================================================
    # TRAIN
    # ========================================================

    model.train(

        data=TRAIN_YAML.as_posix(),

        epochs=50,
        imgsz=640,
        batch=64,

        cache="ram",
        workers=16,

        deterministic=False,
        amp=False,

        patience=20,

        device=0,

        project=RUNS_DIR.as_posix(),
        name=RUN_NAME,

        cls=0.7,
        box=7.5,
        dfl=1.5,

        save=True,
        val=True,

        verbose=True,
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    RUN_DIR = RUNS_DIR / RUN_NAME
    BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"

    print("\n========================================")
    print("          TRAINING COMPLETED")
    print("========================================")

    print(f"\nRun directory:")
    print(RUN_DIR)

    print(f"\nBest weights:")
    print(BEST_WEIGHTS)


    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(
            f"\nbest.pt was not created:\n{BEST_WEIGHTS}"
        )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\nLoading best.pt for TEST evaluation...")

    best_model = YOLO(
        BEST_WEIGHTS.as_posix()
    )

    print("best.pt loaded successfully.")


    # ========================================================
    # TEST SET EVALUATION
    # ========================================================

    print("\n========================================")
    print("          TEST SET EVALUATION")
    print("========================================")

    metrics = best_model.val(

        # IMPORTANT:
        # Original data.yaml is used here.
        # Oversampling is ONLY for training.
        data=DATA_YAML.as_posix(),

        split="test",

        imgsz=832,
        batch=64,

        device=0,
        workers=8,

        verbose=True,
    )


    # ========================================================
    # FINAL METRICS
    # ========================================================

    print("\n========================================")
    print("           FINAL TEST METRICS")
    print("========================================")

    print(f"\nmAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")

    if hasattr(metrics.box, "map75"):
        print(f"mAP75    : {metrics.box.map75:.4f}")

    print("\nPer-class mAP50-95:")
    print(metrics.box.maps)

    print("\n========================================")
    print("       TRAINING & TEST COMPLETED")
    print("========================================")
