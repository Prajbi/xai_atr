import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "configs")
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "classification")
)

import classification_config as config

from aircraft_dataset import AircraftDataset


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(config.RANDOM_SEED)

np.random.seed(config.RANDOM_SEED)

torch.manual_seed(config.RANDOM_SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        config.RANDOM_SEED
    )


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print("=" * 70)
print("DEVICE")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(
    config.CSV_FILE
)

print("\nDataset:", df.shape)


# ============================================================
# SPLITS
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "validation"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print("\nSplit sizes:")

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)


# ============================================================
# CLASS MAPPING
# ============================================================

classes = sorted(
    train_df["class"]
    .astype(str)
    .unique()
)

class_to_idx = {
    cls: idx
    for idx, cls in enumerate(classes)
}

idx_to_class = {
    idx: cls
    for cls, idx in class_to_idx.items()
}

num_classes = len(classes)

print(
    "\nNumber of classes:",
    num_classes
)


# ============================================================
# VERIFY VALIDATION / TEST CLASSES
# ============================================================

print("\nChecking class coverage...")

val_classes = set(
    val_df["class"]
    .astype(str)
    .unique()
)

test_classes = set(
    test_df["class"]
    .astype(str)
    .unique()
)

missing_val = (
    set(classes) - val_classes
)

missing_test = (
    set(classes) - test_classes
)

print(
    "Classes missing from validation:",
    len(missing_val)
)

if missing_val:

    print(
        sorted(missing_val)
    )

print(
    "Classes missing from test:",
    len(missing_test)
)

if missing_test:

    print(
        sorted(missing_test)
    )


# ============================================================
# TRAIN CLASS DISTRIBUTION
# ============================================================

train_class_counts = (
    train_df["class"]
    .astype(str)
    .value_counts()
)


print("\n")
print("=" * 70)
print("TRAIN CLASS DISTRIBUTION")
print("=" * 70)

print(
    "Maximum samples:",
    train_class_counts.max()
)

print(
    "Minimum samples:",
    train_class_counts.min()
)

print(
    "Median samples:",
    train_class_counts.median()
)

print(
    "Imbalance ratio:",
    f"{train_class_counts.max() / train_class_counts.min():.2f}"
)


# ============================================================
# EFFECTIVE-NUMBER CLASS WEIGHTS
# ============================================================

BETA = 0.999


print("\n")
print("=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

print(
    "Weighting method:",
    "Effective Number of Samples"
)

print(
    "Beta:",
    BETA
)


class_weights = []


for cls in classes:

    n = int(
        train_class_counts[cls]
    )

    effective_num = (
        1.0 -
        (BETA ** n)
    )

    weight = (
        (1.0 - BETA)
        / effective_num
    )

    class_weights.append(
        weight
    )


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
)


# ------------------------------------------------------------
# Normalize weights
# ------------------------------------------------------------

class_weights = (
    class_weights /
    class_weights.mean()
)


print("\nClass weight statistics:")

print(
    "Minimum weight:",
    f"{class_weights.min().item():.6f}"
)

print(
    "Maximum weight:",
    f"{class_weights.max().item():.6f}"
)

print(
    "Mean weight:",
    f"{class_weights.mean().item():.6f}"
)


# ------------------------------------------------------------
# Display weights for rarest / most common classes
# ------------------------------------------------------------

weight_table = pd.DataFrame({

    "class": classes,

    "samples": [
        int(train_class_counts[c])
        for c in classes
    ],

    "weight": class_weights.numpy(),

})

weight_table = (
    weight_table
    .sort_values(
        "samples",
        ascending=False
    )
)


print("\nTop 10 common classes:")

print(
    weight_table.head(10)
    .to_string(index=False)
)


print("\nBottom 10 rare classes:")

print(
    weight_table.tail(10)
    .sort_values(
        "samples"
    )
    .to_string(index=False)
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE,
        )
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),

])


eval_transform = transforms.Compose([

    transforms.Resize(
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE,
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),

])


# ============================================================
# DATASETS
# ============================================================

print("\n")
print("=" * 70)
print("CREATING DATASETS")
print("=" * 70)


train_dataset = AircraftDataset(
    train_df,
    config.IMAGE_DIR,
    train_transform,
    class_to_idx,
)


val_dataset = AircraftDataset(
    val_df,
    config.IMAGE_DIR,
    eval_transform,
    class_to_idx,
)


test_dataset = AircraftDataset(
    test_df,
    config.IMAGE_DIR,
    eval_transform,
    class_to_idx,
)


# ============================================================
# DATALOADERS
# ============================================================

print("\nCreating dataloaders...")


train_loader = DataLoader(

    train_dataset,

    batch_size=config.BATCH_SIZE,

    shuffle=True,

    num_workers=config.NUM_WORKERS,

    pin_memory=(
        device.type == "cuda"
    ),

    persistent_workers=(
        config.NUM_WORKERS > 0
    ),

)


val_loader = DataLoader(

    val_dataset,

    batch_size=config.BATCH_SIZE,

    shuffle=False,

    num_workers=config.NUM_WORKERS,

    pin_memory=(
        device.type == "cuda"
    ),

    persistent_workers=(
        config.NUM_WORKERS > 0
    ),

)


test_loader = DataLoader(

    test_dataset,

    batch_size=config.BATCH_SIZE,

    shuffle=False,

    num_workers=config.NUM_WORKERS,

    pin_memory=(
        device.type == "cuda"
    ),

    persistent_workers=(
        config.NUM_WORKERS > 0
    ),

)


# ============================================================
# MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING RESNET-50")
print("=" * 70)


model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)


# ------------------------------------------------------------
# Replace classifier
# ------------------------------------------------------------

num_features = (
    model.fc.in_features
)


model.fc = nn.Sequential(

    nn.Dropout(
        p=config.DROPOUT
    ),

    nn.Linear(
        num_features,
        num_classes
    ),

)


model = model.to(device)


# ============================================================
# LOSS
# ============================================================

class_weights = (
    class_weights.to(device)
)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


print("\nUsing weighted CrossEntropyLoss.")


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=config.LEARNING_RATE,

    weight_decay=config.WEIGHT_DECAY,

)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2,

    )
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

model_dir = (
    config.MODEL_DIR /
    "weighted"
)


result_dir = (
    config.RESULT_DIR
)


model_dir.mkdir(
    parents=True,
    exist_ok=True,
)


result_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SAVE CLASS WEIGHTS
# ============================================================

weight_table.to_csv(

    result_dir /
    "weighted_class_weights.csv",

    index=False,

)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    predictions = []

    targets = []


    progress = tqdm(

        train_loader,

        desc="Training",

    )


    for images, labels in progress:

        images = images.to(

            device,

            non_blocking=True,

        )

        labels = labels.to(

            device,

            non_blocking=True,

        )


        optimizer.zero_grad(

            set_to_none=True

        )


        outputs = model(images)


        loss = criterion(

            outputs,

            labels,

        )


        loss.backward()


        optimizer.step()


        running_loss += (

            loss.item()
            * images.size(0)

        )


        preds = outputs.argmax(

            dim=1

        )


        predictions.extend(

            preds.detach()
            .cpu()
            .numpy()

        )


        targets.extend(

            labels.detach()
            .cpu()
            .numpy()

        )


        progress.set_postfix(

            loss=f"{loss.item():.4f}"

        )


    epoch_loss = (

        running_loss
        / len(train_loader.dataset)

    )


    accuracy = accuracy_score(

        targets,

        predictions,

    )


    macro_f1 = f1_score(

        targets,

        predictions,

        average="macro",

        zero_division=0,

    )


    return (

        epoch_loss,

        accuracy,

        macro_f1,

    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    running_loss = 0.0

    predictions = []

    targets = []


    for images, labels in loader:

        images = images.to(

            device,

            non_blocking=True,

        )

        labels = labels.to(

            device,

            non_blocking=True,

        )


        outputs = model(images)


        loss = criterion(

            outputs,

            labels,

        )


        running_loss += (

            loss.item()
            * images.size(0)

        )


        preds = outputs.argmax(

            dim=1

        )


        predictions.extend(

            preds.cpu()
            .numpy()

        )


        targets.extend(

            labels.cpu()
            .numpy()

        )


    epoch_loss = (

        running_loss
        / len(loader.dataset)

    )


    accuracy = accuracy_score(

        targets,

        predictions,

    )


    macro_precision = precision_score(

        targets,

        predictions,

        average="macro",

        zero_division=0,

    )


    macro_recall = recall_score(

        targets,

        predictions,

        average="macro",

        zero_division=0,

    )


    macro_f1 = f1_score(

        targets,

        predictions,

        average="macro",

        zero_division=0,

    )


    weighted_f1 = f1_score(

        targets,

        predictions,

        average="weighted",

        zero_division=0,

    )


    return {

        "loss": epoch_loss,

        "accuracy": accuracy,

        "macro_precision": macro_precision,

        "macro_recall": macro_recall,

        "macro_f1": macro_f1,

        "weighted_f1": weighted_f1,

        "predictions": predictions,

        "targets": targets,

    }


# ============================================================
# TRAINING
# ============================================================

best_val_f1 = -1.0

epochs_without_improvement = 0

history = []


print("\n")
print("=" * 70)
print("STARTING WEIGHTED TRAINING")
print("=" * 70)


for epoch in range(

    config.NUM_EPOCHS

):


    print("\n")

    print("=" * 70)

    print(

        f"Epoch "
        f"{epoch + 1}/"
        f"{config.NUM_EPOCHS}"

    )

    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    (
        train_loss,
        train_acc,
        train_f1,
    ) = train_one_epoch()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_metrics = evaluate(

        val_loader

    )


    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler.step(

        val_metrics["macro_f1"]

    )


    current_lr = (

        optimizer.param_groups[0]["lr"]

    )


    # --------------------------------------------------------
    # PRINT TRAIN RESULTS
    # --------------------------------------------------------

    print("\nTRAIN")


    print(

        f"Loss     : "
        f"{train_loss:.4f}"

    )


    print(

        f"Accuracy : "
        f"{train_acc:.4f}"

    )


    print(

        f"Macro F1 : "
        f"{train_f1:.4f}"

    )


    # --------------------------------------------------------
    # PRINT VALIDATION RESULTS
    # --------------------------------------------------------

    print("\nVALIDATION")


    print(

        f"Loss     : "
        f"{val_metrics['loss']:.4f}"

    )


    print(

        f"Accuracy : "
        f"{val_metrics['accuracy']:.4f}"

    )


    print(

        f"Macro Precision : "
        f"{val_metrics['macro_precision']:.4f}"

    )


    print(

        f"Macro Recall    : "
        f"{val_metrics['macro_recall']:.4f}"

    )


    print(

        f"Macro F1        : "
        f"{val_metrics['macro_f1']:.4f}"

    )


    print(

        f"Weighted F1     : "
        f"{val_metrics['weighted_f1']:.4f}"

    )


    print(

        f"Learning rate   : "
        f"{current_lr:.6f}"

    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history.append({

        "epoch":
            epoch + 1,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_acc,

        "train_macro_f1":
            train_f1,

        "val_loss":
            val_metrics["loss"],

        "val_accuracy":
            val_metrics["accuracy"],

        "val_macro_precision":
            val_metrics["macro_precision"],

        "val_macro_recall":
            val_metrics["macro_recall"],

        "val_macro_f1":
            val_metrics["macro_f1"],

        "val_weighted_f1":
            val_metrics["weighted_f1"],

        "learning_rate":
            current_lr,

    })


    history_df = pd.DataFrame(

        history

    )


    history_df.to_csv(

        result_dir /
        "weighted_history.csv",

        index=False,

    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if (

        val_metrics["macro_f1"]
        > best_val_f1

    ):

        best_val_f1 = (

            val_metrics["macro_f1"]

        )


        epochs_without_improvement = 0


        checkpoint = {

            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "val_macro_f1":
                best_val_f1,

            "class_to_idx":
                class_to_idx,

            "idx_to_class":
                idx_to_class,

            "num_classes":
                num_classes,

            "image_size":
                config.IMAGE_SIZE,

            "weighting_method":
                "effective_number",

            "beta":
                BETA,

            "class_weights":
                class_weights.detach()
                .cpu(),

        }


        torch.save(

            checkpoint,

            model_dir /
            "resnet50_weighted_best.pth",

        )


        print(

            "\n✓ Best weighted model saved."

        )


    else:

        epochs_without_improvement += 1


        print(

            "\nNo validation "
            "Macro-F1 improvement."

        )


        print(

            "Early stopping counter:",

            f"{epochs_without_improvement}/"
            f"{config.EARLY_STOPPING_PATIENCE}",

        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (

        epochs_without_improvement
        >= config.EARLY_STOPPING_PATIENCE

    ):

        print(

            "\n✓ Early stopping triggered."

        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n")

print("=" * 70)

print("LOADING BEST WEIGHTED MODEL")

print("=" * 70)


checkpoint = torch.load(

    model_dir /
    "resnet50_weighted_best.pth",

    map_location=device,

)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]

)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating on TEST set...")


test_metrics = evaluate(

    test_loader

)


print("\n")

print("=" * 70)

print("FINAL WEIGHTED TEST RESULTS")

print("=" * 70)


print(

    f"Accuracy        : "
    f"{test_metrics['accuracy']:.4f}"

)


print(

    f"Macro Precision : "
    f"{test_metrics['macro_precision']:.4f}"

)


print(

    f"Macro Recall    : "
    f"{test_metrics['macro_recall']:.4f}"

)


print(

    f"Macro F1        : "
    f"{test_metrics['macro_f1']:.4f}"

)


print(

    f"Weighted F1     : "
    f"{test_metrics['weighted_f1']:.4f}"

)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions = []


for target, prediction in zip(

    test_metrics["targets"],

    test_metrics["predictions"],

):


    test_predictions.append({

        "true_idx":
            target,

        "predicted_idx":
            prediction,

        "true_class":
            idx_to_class[target],

        "predicted_class":
            idx_to_class[prediction],

    })


test_predictions_df = pd.DataFrame(

    test_predictions

)


test_predictions_df.to_csv(

    result_dir /
    "weighted_test_predictions.csv",

    index=False,

)


# ============================================================
# SAVE TEST SUMMARY
# ============================================================

test_summary = pd.DataFrame([{

    "accuracy":
        test_metrics["accuracy"],

    "macro_precision":
        test_metrics["macro_precision"],

    "macro_recall":
        test_metrics["macro_recall"],

    "macro_f1":
        test_metrics["macro_f1"],

    "weighted_f1":
        test_metrics["weighted_f1"],

}])


test_summary.to_csv(

    result_dir /
    "weighted_test_metrics.csv",

    index=False,

)


# ============================================================
# SAVE PER-CLASS METRICS
# ============================================================

from sklearn.metrics import classification_report


report = classification_report(

    test_metrics["targets"],

    test_metrics["predictions"],

    labels=list(range(num_classes)),

    target_names=classes,

    output_dict=True,

    zero_division=0,

)


report_df = pd.DataFrame(

    report

).transpose()


report_df.to_csv(

    result_dir /
    "weighted_per_class_metrics.csv",

)


# ============================================================
# FINAL
# ============================================================

print("\n")

print("=" * 70)

print("WEIGHTED TRAINING COMPLETE")

print("=" * 70)


print(

    "Best validation Macro-F1:",

    f"{best_val_f1:.4f}"

)


print(

    "Test Macro-F1:",

    f"{test_metrics['macro_f1']:.4f}"

)


print(

    "Test Accuracy:",

    f"{test_metrics['accuracy']:.4f}"

)


print(

    "\nModel:",

    model_dir /
    "resnet50_weighted_best.pth"

)


print(

    "\nResults:",

    result_dir

)
