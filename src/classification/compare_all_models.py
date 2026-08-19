from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "metrics"
)

OUTPUT_DIR = METRICS_DIR


# ============================================================
# FILES
# ============================================================

FILES = {
    "baseline": (
        PROJECT_ROOT
        / "outputs"
        / "baseline_test_predictions.csv"
    ),

    "weighted": (
        PROJECT_ROOT
        / "outputs"
        / "weighted_test_predictions.csv"
    ),

    "sampler": (
        PROJECT_ROOT
        / "outputs"
        / "classification"
        / "metrics"
        / "sampler_test_predictions.csv"
    ),
}
# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 80)
print("LOADING MODEL PREDICTIONS")
print("=" * 80)

predictions = {}

for model_name, path in FILES.items():

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    df = pd.read_csv(path)

    predictions[model_name] = df

    print(
        f"{model_name.capitalize():10s}: "
        f"{len(df)} predictions"
    )


# ============================================================
# CHECK TEST SET CONSISTENCY
# ============================================================

baseline = predictions["baseline"]
weighted = predictions["weighted"]
sampler = predictions["sampler"]


if not (
    baseline["true_class"].equals(
        weighted["true_class"]
    )
    and
    baseline["true_class"].equals(
        sampler["true_class"]
    )
):

    raise ValueError(
        "Test labels are not identical "
        "across models."
    )


print(
    "\n✓ All models use the same test labels."
)


# ============================================================
# CLASS LIST
# ============================================================

classes = sorted(
    baseline["true_class"]
    .astype(str)
    .unique()
)


print(
    f"Number of test classes: {len(classes)}"
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

all_rows = []


for cls in classes:

    row = {
        "class": cls,
    }

    for model_name, df in predictions.items():

        y_true = (
            df["true_class"]
            .astype(str)
        )

        y_pred = (
            df["predicted_class"]
            .astype(str)
        )

        labels = [cls]

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                average=None,
                zero_division=0,
            )
        )

        row[
            f"{model_name}_precision"
        ] = precision[0]

        row[
            f"{model_name}_recall"
        ] = recall[0]

        row[
            f"{model_name}_f1"
        ] = f1[0]

        row[
            f"{model_name}_support"
        ] = support[0]

    all_rows.append(row)


result = pd.DataFrame(
    all_rows
)


# ============================================================
# F1 CHANGES
# ============================================================

result["weighted_vs_baseline_f1"] = (
    result["weighted_f1"]
    - result["baseline_f1"]
)

result["sampler_vs_baseline_f1"] = (
    result["sampler_f1"]
    - result["baseline_f1"]
)

result["sampler_vs_weighted_f1"] = (
    result["sampler_f1"]
    - result["weighted_f1"]
)


# ============================================================
# RECALL CHANGES
# ============================================================

result["weighted_vs_baseline_recall"] = (
    result["weighted_recall"]
    - result["baseline_recall"]
)

result["sampler_vs_baseline_recall"] = (
    result["sampler_recall"]
    - result["baseline_recall"]
)

result["sampler_vs_weighted_recall"] = (
    result["sampler_recall"]
    - result["weighted_recall"]
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR
    / "baseline_weighted_sampler_per_class.csv"
)

result.to_csv(
    output_file,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("THREE-MODEL PER-CLASS COMPARISON")
print("=" * 80)

print(
    f"Classes: {len(result)}"
)


# ============================================================
# BEST MODEL PER CLASS
# ============================================================

def best_model(row):

    scores = {
        "baseline": row["baseline_f1"],
        "weighted": row["weighted_f1"],
        "sampler": row["sampler_f1"],
    }

    return max(
        scores,
        key=scores.get
    )


result["best_model"] = result.apply(
    best_model,
    axis=1,
)


print("\nBest model per class:")

print(
    result["best_model"]
    .value_counts()
    .to_string()
)


# ============================================================
# SAMPLER IMPROVEMENTS
# ============================================================

print("\n")
print("=" * 80)
print("TOP 20 SAMPLER F1 IMPROVEMENTS VS BASELINE")
print("=" * 80)

cols = [
    "class",
    "baseline_f1",
    "sampler_f1",
    "sampler_vs_baseline_f1",
]

print(
    result
    .sort_values(
        "sampler_vs_baseline_f1",
        ascending=False,
    )
    [cols]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAMPLER DROPS
# ============================================================

print("\n")
print("=" * 80)
print("TOP 20 SAMPLER F1 DROPS VS BASELINE")
print("=" * 80)

print(
    result
    .sort_values(
        "sampler_vs_baseline_f1",
        ascending=True,
    )
    [cols]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAMPLER VS WEIGHTED
# ============================================================

print("\n")
print("=" * 80)
print("TOP 20 SAMPLER IMPROVEMENTS VS WEIGHTED")
print("=" * 80)

cols2 = [
    "class",
    "weighted_f1",
    "sampler_f1",
    "sampler_vs_weighted_f1",
]

print(
    result
    .sort_values(
        "sampler_vs_weighted_f1",
        ascending=False,
    )
    [cols2]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAMPLER VS WEIGHTED DROPS
# ============================================================

print("\n")
print("=" * 80)
print("TOP 20 SAMPLER DROPS VS WEIGHTED")
print("=" * 80)

print(
    result
    .sort_values(
        "sampler_vs_weighted_f1",
        ascending=True,
    )
    [cols2]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "Sampler improved over baseline:",
    (
        result["sampler_vs_baseline_f1"] > 0
    ).sum()
)

print(
    "Sampler unchanged vs baseline:",
    (
        result["sampler_vs_baseline_f1"] == 0
    ).sum()
)

print(
    "Sampler decreased vs baseline:",
    (
        result["sampler_vs_baseline_f1"] < 0
    ).sum()
)

print(
    "\nSampler improved over weighted:",
    (
        result["sampler_vs_weighted_f1"] > 0
    ).sum()
)

print(
    "Sampler unchanged vs weighted:",
    (
        result["sampler_vs_weighted_f1"] == 0
    ).sum()
)

print(
    "Sampler decreased vs weighted:",
    (
        result["sampler_vs_weighted_f1"] < 0
    ).sum()
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("SAVED")
print("=" * 80)

print(output_file)
