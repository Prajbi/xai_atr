from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "labels_with_split.csv"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "metrics"
)


# ============================================================
# LOAD DATA
# ============================================================

labels = pd.read_csv(CSV_FILE)

train = labels[
    labels["split"] == "train"
].copy()


# ============================================================
# TRAINING CLASS COUNTS
# ============================================================

class_counts = (
    train["class"]
    .astype(str)
    .value_counts()
)


# ============================================================
# DEFINE HEAD / MEDIUM / TAIL
# ============================================================

def assign_group(count):

    if count >= 500:
        return "Head"

    elif count >= 100:
        return "Medium"

    else:
        return "Tail"


frequency = pd.DataFrame({
    "class": class_counts.index,
    "train_count": class_counts.values,
})

frequency["group"] = (
    frequency["train_count"]
    .apply(assign_group)
)


# ============================================================
# LOAD THREE-MODEL COMPARISON
# ============================================================

comparison_file = (
    METRICS_DIR
    / "baseline_weighted_sampler_per_class.csv"
)

df = pd.read_csv(
    comparison_file
)


# ============================================================
# MERGE
# ============================================================

df["class"] = (
    df["class"]
    .astype(str)
)

frequency["class"] = (
    frequency["class"]
    .astype(str)
)

df = df.merge(
    frequency,
    on="class",
    how="left",
)


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

print("=" * 80)
print("THREE-MODEL HEAD / MEDIUM / TAIL ANALYSIS")
print("=" * 80)

print("\nClass distribution:")

print(
    df["group"]
    .value_counts()
    .to_string()
)


# ============================================================
# GROUP SUMMARY
# ============================================================

summary_rows = []


for group in [
    "Head",
    "Medium",
    "Tail",
]:

    subset = df[
        df["group"] == group
    ]

    summary_rows.append({

        "group": group,

        "classes":
            len(subset),

        "mean_train_count":
            subset["train_count"].mean(),

        "baseline_f1":
            subset["baseline_f1"].mean(),

        "weighted_f1":
            subset["weighted_f1"].mean(),

        "sampler_f1":
            subset["sampler_f1"].mean(),

        "baseline_recall":
            subset["baseline_recall"].mean(),

        "weighted_recall":
            subset["weighted_recall"].mean(),

        "sampler_recall":
            subset["sampler_recall"].mean(),

    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# CHANGES
# ============================================================

summary["weighted_vs_baseline_f1"] = (
    summary["weighted_f1"]
    - summary["baseline_f1"]
)

summary["sampler_vs_baseline_f1"] = (
    summary["sampler_f1"]
    - summary["baseline_f1"]
)

summary["sampler_vs_weighted_f1"] = (
    summary["sampler_f1"]
    - summary["weighted_f1"]
)

summary["weighted_vs_baseline_recall"] = (
    summary["weighted_recall"]
    - summary["baseline_recall"]
)

summary["sampler_vs_baseline_recall"] = (
    summary["sampler_recall"]
    - summary["baseline_recall"]
)

summary["sampler_vs_weighted_recall"] = (
    summary["sampler_recall"]
    - summary["weighted_recall"]
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
# CLASS-LEVEL WIN COUNTS
# ============================================================

print("\n")
print("=" * 80)
print("BEST MODEL BY FREQUENCY GROUP")
print("=" * 80)


for group in [
    "Head",
    "Medium",
    "Tail",
]:

    subset = df[
        df["group"] == group
    ].copy()

    subset["best"] = subset.apply(
        lambda row: max(
            [
                ("baseline", row["baseline_f1"]),
                ("weighted", row["weighted_f1"]),
                ("sampler", row["sampler_f1"]),
            ],
            key=lambda x: x[1],
        )[0],
        axis=1,
    )

    print(f"\n{group}")

    print(
        subset["best"]
        .value_counts()
        .to_string()
    )


# ============================================================
# TAIL CLASSES
# ============================================================

print("\n")
print("=" * 80)
print("TAIL CLASS PERFORMANCE")
print("=" * 80)

tail = df[
    df["group"] == "Tail"
].copy()

tail_columns = [
    "class",
    "train_count",
    "baseline_f1",
    "weighted_f1",
    "sampler_f1",
    "baseline_recall",
    "weighted_recall",
    "sampler_recall",
]

print(
    tail
    .sort_values(
        "train_count"
    )
    [tail_columns]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SAVE
# ============================================================

summary_file = (
    METRICS_DIR
    / "three_model_head_medium_tail_summary.csv"
)

detail_file = (
    METRICS_DIR
    / "three_model_head_medium_tail_per_class.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)

df.to_csv(
    detail_file,
    index=False,
)


print("\n")
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(summary_file)
print(detail_file)
