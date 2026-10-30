from pathlib import Path
from collections import Counter
import csv


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(
    r"D:\Blood_Smear_AI\data\rbc_morphology\prepared"
)

METADATA = Path(
    r"D:\Blood_Smear_AI\data\rbc_morphology\metadata"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

SPLITS = [
    "train",
    "val",
    "test"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("RBC MORPHOLOGY - PREPARED DATASET ANALYSIS")
print("=" * 75)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET}"
    )


# ============================================================
# COUNT IMAGES
# ============================================================

distribution = {}

all_classes = set()

for split in SPLITS:

    split_dir = DATASET / split

    if not split_dir.exists():

        raise FileNotFoundError(
            f"Missing split:\n{split_dir}"
        )

    distribution[split] = {}

    for class_dir in sorted(
        split_dir.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        count = sum(
            1
            for file in class_dir.rglob("*")
            if (
                file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        distribution[split][class_name] = count

        all_classes.add(class_name)


classes = sorted(all_classes)


# ============================================================
# DISPLAY DISTRIBUTION
# ============================================================

print("\nCLASS DISTRIBUTION")
print("-" * 75)

print(
    f"{'Class':<25}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
    f"{'Total':>10}"
    f"{'Train %':>10}"
)

print("-" * 75)


for class_name in classes:

    train_count = distribution["train"].get(
        class_name,
        0
    )

    val_count = distribution["val"].get(
        class_name,
        0
    )

    test_count = distribution["test"].get(
        class_name,
        0
    )

    total = (
        train_count
        + val_count
        + test_count
    )

    train_total = sum(
        distribution["train"].values()
    )

    train_percent = (
        train_count /
        train_total *
        100
    )

    print(
        f"{class_name:<25}"
        f"{train_count:>10}"
        f"{val_count:>10}"
        f"{test_count:>10}"
        f"{total:>10}"
        f"{train_percent:>9.2f}%"
    )


# ============================================================
# TOTALS
# ============================================================

train_total = sum(
    distribution["train"].values()
)

val_total = sum(
    distribution["val"].values()
)

test_total = sum(
    distribution["test"].values()
)

grand_total = (
    train_total
    + val_total
    + test_total
)

print("-" * 75)

print(
    f"{'TOTAL':<25}"
    f"{train_total:>10}"
    f"{val_total:>10}"
    f"{test_total:>10}"
    f"{grand_total:>10}"
)


# ============================================================
# IMBALANCE ANALYSIS
# ============================================================

train_counts = {
    class_name:
        distribution["train"].get(
            class_name,
            0
        )
    for class_name in classes
}

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
    f"\nSmallest class : "
    f"{smallest_class}"
)

print(
    f"Smallest count : "
    f"{smallest_count}"
)

print(
    f"\nImbalance ratio : "
    f"{imbalance_ratio:.2f}x"
)


# ============================================================
# SIMPLE CLASS WEIGHTS
#
# weight = total / (number_of_classes * class_count)
# ============================================================

num_classes = len(classes)

class_weights = {}

for class_name in classes:

    count = train_counts[class_name]

    weight = (
        train_total /
        (
            num_classes *
            count
        )
    )

    class_weights[class_name] = weight


print("\n")
print("=" * 75)
print("SUGGESTED CLASS WEIGHTS")
print("=" * 75)

for class_name in classes:

    print(
        f"{class_name:<25}"
        f"{class_weights[class_name]:.4f}"
    )


# ============================================================
# CHECK ALL SPLITS
# ============================================================

print("\n")
print("=" * 75)
print("SPLIT VALIDATION")
print("=" * 75)

for class_name in classes:

    missing = []

    for split in SPLITS:

        if distribution[split].get(
            class_name,
            0
        ) == 0:

            missing.append(split)

    if missing:

        print(
            f"WARNING: {class_name} "
            f"missing from {missing}"
        )


print("\nAll classes should appear in train, val and test.")


# ============================================================
# SAVE CSV
# ============================================================

output_file = (
    METADATA /
    "rbc_training_distribution.csv"
)

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "class",
            "train",
            "val",
            "test",
            "total",
            "train_percent",
            "class_weight"
        ]
    )

    for class_name in classes:

        train_count = distribution[
            "train"
        ].get(class_name, 0)

        val_count = distribution[
            "val"
        ].get(class_name, 0)

        test_count = distribution[
            "test"
        ].get(class_name, 0)

        total = (
            train_count
            + val_count
            + test_count
        )

        train_percent = (
            train_count /
            train_total *
            100
        )

        writer.writerow(
            [
                class_name,
                train_count,
                val_count,
                test_count,
                total,
                round(train_percent, 4),
                round(
                    class_weights[class_name],
                    6
                )
            ]
        )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"\nReport saved to:\n{output_file}"
)

print("\nOriginal dataset was NOT modified.")