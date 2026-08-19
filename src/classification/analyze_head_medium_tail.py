from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    ROOT
    / "data"
    / "raw"
    / "labels_with_split.csv"
)

COMPARISON_FILE = (
    ROOT
    / "outputs"
    / "classification"
    / "metrics"
    / "baseline_vs_weighted_per_class.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "classification"
    / "metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

train_df = df[
    df["split"] == "train"
].copy()

comparison = pd.read_csv(
    COMPARISON_FILE
)


# ============================================================
# TRAINING CLASS COUNTS
# ============================================================

train_counts = (
    train_df["class"]
    .astype(str)
    .value_counts()
    .rename("train_count")
    .reset_index()
)

train_counts.columns = [
    "class",
    "train_count",
]


# ============================================================
# MERGE
# ============================================================

# Keep only actual aircraft classes
comparison = comparison[
    comparison["class"].isin(
        train_counts["class"]
    )
].copy()

analysis = comparison.merge(
    train_counts,
    on="class",
    how="inner",
)

# ============================================================
# FREQUENCY GROUP
# ============================================================

def frequency_group(n):

    if n >= 500:
        return "Head"

    elif n >= 100:
        return "Medium"

    else:
        return "Tail"


analysis["group"] = (
    analysis["train_count"]
    .apply(frequency_group)
)


# ============================================================
# SAVE COMPLETE ANALYSIS
# ============================================================

output_file = (
    OUTPUT_DIR
    / "head_medium_tail_comparison.csv"
)

analysis.to_csv(
    output_file,
    index=False,
)


# ============================================================
# GROUP SUMMARY
# ============================================================

print("=" * 80)
print("HEAD / MEDIUM / TAIL ANALYSIS")
print("=" * 80)

print(
    "\nClass counts:"
)

print(
    analysis["group"]
    .value_counts()
    .to_string()
)


# ============================================================
# METRIC SUMMARY
# ============================================================

summary = (
    analysis
    .groupby("group")
    .agg(
        classes=("class", "count"),

        mean_train_count=(
            "train_count",
            "mean",
        ),

        baseline_f1=(
            "baseline_f1",
            "mean",
        ),

        weighted_f1=(
            "weighted_f1",
            "mean",
        ),

        f1_change=(
            "f1_change",
            "mean",
        ),

        baseline_recall=(
            "baseline_recall",
            "mean",
        ),

        weighted_recall=(
            "weighted_recall",
            "mean",
        ),

        recall_change=(
            "recall_change",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# ORDER
# ============================================================

order = {
    "Head": 0,
    "Medium": 1,
    "Tail": 2,
}

summary["order"] = (
    summary["group"]
    .map(order)
)

summary = (
    summary
    .sort_values("order")
    .drop(columns="order")
)


# ============================================================
# PRINT
# ============================================================

print("\n")
print("=" * 80)
print("PER-FREQUENCY PERFORMANCE")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# IMPROVEMENT COUNTS
# ============================================================

print("\n")
print("=" * 80)
print("IMPROVEMENT / DECREASE COUNTS")
print("=" * 80)

for group in [
    "Head",
    "Medium",
    "Tail",
]:

    subset = analysis[
        analysis["group"] == group
    ]

    improved = (
        subset["f1_change"] > 0
    ).sum()

    unchanged = (
        subset["f1_change"] == 0
    ).sum()

    decreased = (
        subset["f1_change"] < 0
    ).sum()

    print(
        f"\n{group}"
    )

    print(
        f"  Improved  : {improved}"
    )

    print(
        f"  Unchanged : {unchanged}"
    )

    print(
        f"  Decreased : {decreased}"
    )


# ============================================================
# TAIL CLASSES
# ============================================================

tail = analysis[
    analysis["group"] == "Tail"
].sort_values(
    "train_count"
)


print("\n")
print("=" * 80)
print("TAIL CLASSES")
print("=" * 80)

print(
    tail[
        [
            "class",
            "train_count",
            "test_count",
            "baseline_f1",
            "weighted_f1",
            "f1_change",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    OUTPUT_DIR
    / "head_medium_tail_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)


print("\n")
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    output_file
)

print(
    summary_file
)
