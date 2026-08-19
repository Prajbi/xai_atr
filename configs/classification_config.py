from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = ROOT / "data/raw/labels_with_split.csv"
IMAGE_DIR = ROOT / "data/raw/dataset"

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "outputs"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8

NUM_EPOCHS = 25

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 5

RANDOM_SEED = 42
