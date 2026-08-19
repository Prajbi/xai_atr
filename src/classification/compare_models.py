from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    ROOT /
    "data" /
    "raw" /
    "labels_with_split.csv"
)

BASELINE_FILE = (
    ROOT /
    "outputs" /
    "baseline_test_predictions.csv"
)

WEIGHTED_FILE = (
    ROOT /
    "outputs" /
    "weighted_test_predictions.csv"
)

OUTPUT_DIR = (
    ROOT /
    "outputs" /
    "classification" /
    "metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

labels = pd.read_csv(CSV_FILE)

test_labels = labels[
    labels["split"] == "test"
].copy()

baseline = pd.read_csv(
    BASELINE_FILE
)

weighted = pd.read_csv(
    WEIGHTED_FILE
)


print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    "Test labels:",
    len(test_labels)
)

print(
    "Baseline predictions:",
    len(baseline)
)

print(
    "Weighted predictions:",
    len(weighted)
)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

if len(baseline) != len(weighted):

    raise ValueError(
        "Baseline and weighted prediction files "
        "have different numbers of rows."
    )


# ============================================================
# CLASS COUNTS
# ============================================================

class_counts = (
    test_labels["class"]
    .astype(str)
    .value_counts()
    .rename("test_count")
    .reset_index()
)

class_counts.columns = [
    "class",
    "test_count",
]


# ============================================================
# PER-CLASS FUNCTION
# ============================================================

def calculate_per_class(df):

    y_true = df["true_idx"].to_numpy()

    y_pred = df["predicted_idx"].to_numpy()

    classes = sorted(
        set(y_true) |
        set(y_pred)
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=classes,
            zero_division=0,
        )
    )

    result = pd.DataFrame({

        "class_idx": classes,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "support": support,

    })

    return result


# ============================================================
# CLASS NAME MAPPING
# ============================================================

idx_to_class = {}

for _, row in baseline.iterrows():

    idx = int(row["true_idx"])

    cls = str(row["true_class"])

    idx_to_class[idx] = cls

for _, row in weighted.iterrows():

    idx = int(row["true_idx"])

    cls = str(row["true_class"])

    idx_to_class[idx] = cls


# ============================================================
# CALCULATE
# ============================================================

baseline_metrics = calculate_per_class(
    baseline
)

weighted_metrics = calculate_per_class(
    weighted
)


# ============================================================
# ADD CLASS NAMES
# ============================================================

baseline_metrics["class"] = (
    baseline_metrics["class_idx"]
    .map(idx_to_class)
)

weighted_metrics["class"] = (
    weighted_metrics["class_idx"]
    .map(idx_to_class)
)


# ============================================================
# RENAME
# ============================================================

baseline_metrics = baseline_metrics.rename(
    columns={
        "precision":
            "baseline_precision",

        "recall":
            "baseline_recall",

        "f1":
            "baseline_f1",

        "support":
            "baseline_support",
    }
)

weighted_metrics = weighted_metrics.rename(
    columns={
        "precision":
            "weighted_precision",

        "recall":
            "weighted_recall",

        "f1":
            "weighted_f1",

        "support":
            "weighted_support",
    }
)


# ============================================================
# MERGE
# ============================================================

comparison = baseline_metrics.merge(
    weighted_metrics,
    on=[
        "class_idx",
        "class",
    ],
    how="outer",
)

comparison = comparison.merge(
    class_counts,
    on="class",
    how="left",
)


# ============================================================
# IMPROVEMENTS
# ============================================================

comparison["f1_change"] = (
    comparison["weighted_f1"]
    -
    comparison["baseline_f1"]
)

comparison["recall_change"] = (
    comparison["weighted_recall"]
    -
    comparison["baseline_recall"]
)

comparison["precision_change"] = (
    comparison["weighted_precision"]
    -
    comparison["baseline_precision"]
)


# ============================================================
# SORT BY TRAIN/TEST FREQUENCY
# ============================================================

comparison = comparison.sort_values(
    "test_count"
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR /
    "baseline_vs_weighted_per_class.csv"
)

comparison.to_csv(
    output_file,
    index=False,
)


# ============================================================
# PRINT RARE CLASSES
# ============================================================

print("\n")
print("=" * 80)
print("RARE CLASSES")
print("=" * 80)

rare = comparison[
    comparison["test_count"] <= 10
]

print(
    rare[
        [
            "class",
            "test_count",
            "baseline_f1",
            "weighted_f1",
            "f1_change",
            "baseline_recall",
            "weighted_recall",
            "recall_change",
        ]
    ].to_string(index=False)
)


# ============================================================
# BIGGEST IMPROVEMENTS
# ============================================================

print("\n")
print("=" * 80)
print("TOP 15 F1 IMPROVEMENTS")
print("=" * 80)

improved = comparison.sort_values(
    "f1_change",
    ascending=False,
)

print(
    improved[
        [
            "class",
            "test_count",
            "baseline_f1",
            "weighted_f1",
            "f1_change",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# BIGGEST DROPS
# ============================================================

print("\n")
print("=" * 80)
print("TOP 15 F1 DROPS")
print("=" * 80)

dropped = comparison.sort_values(
    "f1_change",
    ascending=True,
)

print(
    dropped[
        [
            "class",
            "test_count",
            "baseline_f1",
            "weighted_f1",
            "f1_change",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "Classes:",
    len(comparison)
)

print(
    "Improved classes:",
    (comparison["f1_change"] > 0).sum()
)

print(
    "Unchanged classes:",
    (comparison["f1_change"] == 0).sum()
)

print(
    "Decreased classes:",
    (comparison["f1_change"] < 0).sum()
)

print(
    "\nSaved:",
    output_file
)
