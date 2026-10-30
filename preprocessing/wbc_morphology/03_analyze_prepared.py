from pathlib import Path
import csv
import json


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(
    r"D:\Blood_Smear_AI\data\wbc_morphology\prepared"
)

METADATA = Path(
    r"D:\Blood_Smear_AI\data\wbc_morphology\metadata"
)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("WBC MORPHOLOGY - PREPARED DATASET ANALYSIS")
print("=" * 75)


# ============================================================
# DISCOVER CLASSES
# ============================================================

classes = sorted({

    folder.name

    for folder in (
        DATASET / "train"
    ).iterdir()

    if folder.is_dir()
})


num_classes = len(classes)


# ============================================================
# COUNT IMAGES
# ============================================================

distribution = {}

for class_name in classes:

    distribution[class_name] = {}

    for split in SPLITS:

        folder = (
            DATASET /
            split /
            class_name
        )

        count = sum(

            1

            for file in folder.iterdir()

            if (
                file.is_file()
                and
                file.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        distribution[class_name][split] = count


# ============================================================
# TRAINING COUNTS
# ============================================================

train_counts = {

    class_name:
        distribution[class_name]["train"]

    for class_name in classes
}


total_train = sum(
    train_counts.values()
)

largest_class = max(
    train_counts,
    key=train_counts.get
)

smallest_class = min(
    train_counts,
    key=train_counts.get
)

largest_count = train_counts[
    largest_class
]

smallest_count = train_counts[
    smallest_class
]

imbalance_ratio = (
    largest_count /
    smallest_count
)


# ============================================================
# DISPLAY DISTRIBUTION
# ============================================================

print("\n")
print("=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

print(
    f"\n{'Class':<25}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
    f"{'Train %':>12}"
)

print("-" * 72)


for class_name in classes:

    train = distribution[
        class_name
    ]["train"]

    val = distribution[
        class_name
    ]["val"]

    test = distribution[
        class_name
    ]["test"]

    percentage = (
        train /
        total_train *
        100
    )

    print(
        f"{class_name:<25}"
        f"{train:>10}"
        f"{val:>10}"
        f"{test:>10}"
        f"{percentage:>11.2f}%"
    )


print("-" * 72)

print(
    f"{'TOTAL':<25}"
    f"{total_train:>10}"
    f"{sum(distribution[c]['val'] for c in classes):>10}"
    f"{sum(distribution[c]['test'] for c in classes):>10}"
    f"{100:>11.2f}%"
)


# ============================================================
# IMBALANCE
# ============================================================

print("\n")
print("=" * 75)
print("IMBALANCE ANALYSIS")
print("=" * 75)

print(
    f"\nLargest class : "
    f"{largest_class}"
)

print(
    f"Largest count : "
    f"{largest_count}"
)

print(
    f"\nSmallest class: "
    f"{smallest_class}"
)

print(
    f"Smallest count: "
    f"{smallest_count}"
)

print(
    f"\nImbalance ratio: "
    f"{imbalance_ratio:.2f}x"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================
#
# Inverse-frequency weights:
#
#       total / (classes * class_count)
#
# These are diagnostic weights.
# We will NOT automatically use them until
# we inspect the results.
# ============================================================

print("\n")
print("=" * 75)
print("SUGGESTED CLASS WEIGHTS")
print("=" * 75)

class_weights = {}

for class_name in classes:

    count = train_counts[
        class_name
    ]

    weight = (
        total_train /
        (
            num_classes *
            count
        )
    )

    class_weights[
        class_name
    ] = weight

    print(
        f"{class_name:<25}"
        f"{weight:.4f}"
    )


# ============================================================
# MINORITY ANALYSIS
# ============================================================

print("\n")
print("=" * 75)
print("MINORITY CLASS ANALYSIS")
print("=" * 75)

thresholds = [
    0.01,
    0.02,
    0.05
]

for threshold in thresholds:

    minority = [

        class_name

        for class_name in classes

        if (
            train_counts[class_name]
            /
            total_train
        ) < threshold

    ]

    print(
        f"\nClasses below "
        f"{threshold * 100:.0f}%:"
    )

    if minority:

        for class_name in minority:

            percentage = (
                train_counts[class_name]
                /
                total_train
                *
                100
            )

            print(
                f"  {class_name:<25}"
                f"{train_counts[class_name]:>6}"
                f" ({percentage:.2f}%)"
            )

    else:

        print("  None")


# ============================================================
# SPLIT RATIO ANALYSIS
# ============================================================

print("\n")
print("=" * 75)
print("SPLIT ANALYSIS")
print("=" * 75)

for class_name in classes:

    train = distribution[
        class_name
    ]["train"]

    val = distribution[
        class_name
    ]["val"]

    test = distribution[
        class_name
    ]["test"]

    total = (
        train +
        val +
        test
    )

    train_pct = train / total * 100
    val_pct = val / total * 100
    test_pct = test / total * 100

    print(
        f"\n{class_name}"
    )

    print(
        f"  Train: {train_pct:.2f}%"
    )

    print(
        f"  Val  : {val_pct:.2f}%"
    )

    print(
        f"  Test : {test_pct:.2f}%"
    )


# ============================================================
# RECOMMENDATION
# ============================================================

print("\n")
print("=" * 75)
print("TRAINING RECOMMENDATION")
print("=" * 75)

print(
    """
1. Keep the supplied train/validation/test split.
2. Do NOT duplicate minority images aggressively.
3. Use weighted sampling during training.
4. Use moderate augmentation.
5. Monitor Macro F1, not only accuracy.
6. Pay particular attention to the very small classes.
7. Keep validation and test sets untouched.
"""
)


# ============================================================
# SAVE CSV
# ============================================================

csv_file = (
    METADATA /
    "wbc_training_distribution.csv"
)

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class",
        "train",
        "val",
        "test",
        "train_percentage",
        "class_weight"
    ])

    for class_name in classes:

        train = train_counts[
            class_name
        ]

        percentage = (
            train /
            total_train *
            100
        )

        writer.writerow([
            class_name,
            train,
            distribution[class_name]["val"],
            distribution[class_name]["test"],
            percentage,
            class_weights[class_name]
        ])


# ============================================================
# SAVE JSON
# ============================================================

json_file = (
    METADATA /
    "wbc_training_analysis.json"
)

analysis = {

    "num_classes":
        num_classes,

    "total_train":
        total_train,

    "largest_class":
        largest_class,

    "largest_count":
        largest_count,

    "smallest_class":
        smallest_class,

    "smallest_count":
        smallest_count,

    "imbalance_ratio":
        imbalance_ratio,

    "class_weights":
        class_weights,

    "distribution":
        distribution
}


with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        analysis,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"\nCSV:"
    f"\n{csv_file}"
)

print(
    f"\nJSON:"
    f"\n{json_file}"
)

print(
    "\nOriginal dataset was NOT modified."
)