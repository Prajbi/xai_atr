import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from config import (
    CSV_FILE,
    IMAGE_DIR,
    MODEL_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED
)

from aircraft_dataset import AircraftDataset

# ============================================================
# Reproducibility
# ============================================================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================
# Device
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("=" * 70)
print("DEVICE")
print("=" * 70)
print(device)

# ============================================================
# Load CSV & Splits
# ============================================================
df = pd.read_csv(CSV_FILE)
print("\nDataset:", df.shape)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# ============================================================
# Class mapping
# ============================================================
classes = sorted(train_df["class"].unique())
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
num_classes = len(classes)
print("\nNumber of classes:", num_classes)

# ============================================================
# Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# Datasets & Loaders
# ============================================================
train_dataset = AircraftDataset(train_df, IMAGE_DIR, train_transform, class_to_idx)
val_dataset = AircraftDataset(val_df, IMAGE_DIR, eval_transform, class_to_idx)
test_dataset = AircraftDataset(test_df, IMAGE_DIR, eval_transform, class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

# ============================================================
# CLASS WEIGHTS
# ============================================================
class_counts = train_df["class"].value_counts().reindex(classes)
weights = len(train_df) / (num_classes * class_counts.values)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

# ============================================================
# MODEL, LOSS, OPTIMIZER, SCHEDULER
# ============================================================
print("\nLoading ResNet-50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# ============================================================
# RESUME CHECKPOINT LOGIC
# ============================================================
CHECKPOINT_PATH = MODEL_DIR / "resnet50_weighted_best.pth"
start_epoch = 0
best_f1 = 0.0

if CHECKPOINT_PATH.exists():
    print(f"\nLoading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state restored successfully.")
            except Exception as e:
                print(f"Warning: Could not load optimizer state ({e}).")
                
        saved_epoch = checkpoint.get('epoch', 0)
        start_epoch = saved_epoch  # Resume from the next epoch
        best_f1 = checkpoint.get('val_f1', 0.0)
        print(f"Resuming training! Last completed epoch in checkpoint: {saved_epoch} (Best Val F1: {best_f1:.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw state dict. Starting from Epoch 1...")
else:
    print("\nNo checkpoint found! Starting training from scratch.")

# ============================================================
# Training / Validation functions
# ============================================================
def train_one_epoch():
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []

    progress = tqdm(train_loader, desc="Training")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        predictions.extend(preds.detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

        progress.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_dataset)
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return epoch_loss, accuracy, macro_f1

@torch.no_grad()
def evaluate(loader):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return loss, accuracy, macro_f1

# ============================================================
# Training loop
# ============================================================
MODEL_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(start_epoch, NUM_EPOCHS):
    print("\n" + "=" * 70)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("=" * 70)

    train_loss, train_acc, train_f1 = train_one_epoch()
    val_loss, val_acc, val_f1 = evaluate(val_loader)
    
    scheduler.step(val_f1)

    print("\nTraining:")
    print(f"Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | Macro F1: {train_f1:.4f}")

    print("\nValidation:")
    print(f"Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | Macro F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1,
            "class_to_idx": class_to_idx
        }
        torch.save(checkpoint, MODEL_DIR / "resnet50_weighted_best.pth")
        print("\n✓ Best model updated and saved.")

print("\nTraining complete.")
print(f"Best validation Macro-F1: {best_f1:.4f}")
