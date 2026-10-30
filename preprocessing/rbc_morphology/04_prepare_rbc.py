from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import shutil
import json
import csv

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path(
    r"D:\Blood_Smear_AI\data\rbc_morphology\raw\RBC"
)

OUTPUT = Path(
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

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("RBC MORPHOLOGY - LEAKAGE-FREE DATASET PREPARATION")
print("=" * 75)


# ============================================================
# SAFETY CHECKS
# ============================================================

if not SOURCE.exists():

    raise FileNotFoundError(
        f"\nSource dataset not found:\n{SOURCE}"
    )


# The previous run created an incomplete prepared dataset.
# This script intentionally refuses to overwrite existing data.
if OUTPUT.exists():

    existing_files = list(
        OUTPUT.rglob("*")
    )

    if existing_files:

        raise RuntimeError(
            "\nPrepared dataset already contains files.\n"
            f"Location:\n{OUTPUT}\n\n"
            "Delete the incomplete prepared dataset first:\n\n"
            f'Remove-Item "{OUTPUT}" -Recurse -Force\n\n'
            "Then run this script again."
        )


OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

METADATA.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SHA256
# ============================================================

def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# COLLECT IMAGES
# ============================================================

print("\nScanning source dataset...")

records = []

for split in ["train", "val", "test"]:

    split_dir = SOURCE / split

    if not split_dir.exists():

        print(
            f"WARNING: Missing split: {split}"
        )

        continue

    for class_dir in sorted(
        split_dir.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for image in class_dir.rglob("*"):

            if (
                image.is_file()
                and image.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                records.append(
                    {
                        "path": image,
                        "class": class_name,
                        "original_split": split
                    }
                )


print(
    f"Images found: {len(records)}"
)


# ============================================================
# HASH + DUPLICATE GROUPING
# ============================================================

print("\nHashing images and grouping duplicates...")

groups = defaultdict(list)

for i, record in enumerate(
    records,
    1
):

    file_hash = sha256(
        record["path"]
    )

    groups[file_hash].append(
        record
    )

    if i % 1000 == 0:

        print(
            f"Processed "
            f"{i}/{len(records)}"
        )


# ============================================================
# VALIDATE LABEL CONSISTENCY
# ============================================================

print("\nChecking duplicate-label consistency...")

conflicts = []

for file_hash, items in groups.items():

    labels = {
        item["class"]
        for item in items
    }

    if len(labels) > 1:

        conflicts.append(
            {
                "hash": file_hash,
                "labels": sorted(labels),
                "items": items
            }
        )


if conflicts:

    print(
        f"\nERROR: Found "
        f"{len(conflicts)} "
        "cross-label duplicate groups."
    )

    conflict_file = (
        METADATA /
        "prepare_label_conflicts.txt"
    )

    with open(
        conflict_file,
        "w",
        encoding="utf-8"
    ) as f:

        for item in conflicts:

            f.write(
                f"\nHASH: {item['hash']}\n"
            )

            f.write(
                f"LABELS: {item['labels']}\n"
            )

            for record in item["items"]:

                f.write(
                    f"{record['class']} | "
                    f"{record['path']}\n"
                )

    raise RuntimeError(
        "Cross-label conflicts detected.\n"
        f"See:\n{conflict_file}"
    )


print(
    "No cross-label conflicts found."
)


# ============================================================
# CREATE UNIQUE GROUP RECORDS
# ============================================================

unique_groups = []

for file_hash, items in groups.items():

    # All duplicate images have the same class
    class_name = items[0]["class"]

    # Select deterministic representative
    representative = sorted(
        items,
        key=lambda x: str(x["path"])
    )[0]

    unique_groups.append(
        {
            "hash": file_hash,
            "class": class_name,
            "source": representative["path"],
            "duplicate_count": len(items)
        }
    )


print(
    f"\nOriginal images : {len(records)}"
)

print(
    f"Unique groups   : {len(unique_groups)}"
)

print(
    f"Removed duplicate copies : "
    f"{len(records) - len(unique_groups)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = Counter(
    group["class"]
    for group in unique_groups
)

classes = sorted(
    class_counts.keys()
)

print("\nUNIQUE CLASS DISTRIBUTION")
print("-" * 60)

for class_name in classes:

    print(
        f"{class_name:<25}"
        f"{class_counts[class_name]:>8}"
    )


# ============================================================
# FIRST SPLIT
#
# 90% DEVELOPMENT
# 10% TEST
# ============================================================

indices = list(
    range(len(unique_groups))
)

labels = [
    unique_groups[i]["class"]
    for i in indices
]

development_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_RATIO,
    random_state=SEED,
    stratify=labels
)


# ============================================================
# SECOND SPLIT
#
# Development = 90%
#
# From development:
#   88.8889% -> TRAIN
#   11.1111% -> VAL
#
# Overall:
#   80% TRAIN
#   10% VAL
#   10% TEST
# ============================================================

development_labels = [
    unique_groups[i]["class"]
    for i in development_idx
]

val_fraction_of_development = (
    VAL_RATIO /
    (TRAIN_RATIO + VAL_RATIO)
)

train_idx, val_idx = train_test_split(
    development_idx,
    test_size=val_fraction_of_development,
    random_state=SEED,
    stratify=development_labels
)


# ============================================================
# CREATE SPLIT MAP
# ============================================================

split_map = {}

for index in train_idx:

    split_map[index] = "train"

for index in val_idx:

    split_map[index] = "val"

for index in test_idx:

    split_map[index] = "test"


# ============================================================
# VERIFY SPLIT MAP
# ============================================================

if len(split_map) != len(unique_groups):

    raise RuntimeError(
        "\nSplit assignment error.\n"
        f"Unique groups : {len(unique_groups)}\n"
        f"Split entries : {len(split_map)}"
    )


missing_indices = [
    i
    for i in range(len(unique_groups))
    if i not in split_map
]

if missing_indices:

    raise RuntimeError(
        "\nSome unique groups have no split assignment.\n"
        f"Missing groups: {len(missing_indices)}"
    )


# ============================================================
# DISPLAY SPLIT
# ============================================================

print("\n")
print("=" * 75)
print("NEW SPLIT")
print("=" * 75)

print(
    f"TRAIN : {len(train_idx)} "
    f"({len(train_idx) / len(unique_groups) * 100:.2f}%)"
)

print(
    f"VAL   : {len(val_idx)} "
    f"({len(val_idx) / len(unique_groups) * 100:.2f}%)"
)

print(
    f"TEST  : {len(test_idx)} "
    f"({len(test_idx) / len(unique_groups) * 100:.2f}%)"
)

print(
    f"\nSplit assignments verified: "
    f"{len(split_map)} / {len(unique_groups)}"
)


# ============================================================
# PREPARE OUTPUT DIRECTORIES
# ============================================================

print("\nCreating output directories...")

for split in [
    "train",
    "val",
    "test"
]:

    for class_name in classes:

        (
            OUTPUT /
            split /
            class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY IMAGES
# ============================================================

print("\nCopying images...")

manifest = []

# IMPORTANT:
#
# enumerate() starts at 0 here.
#
# This matches the indices used by train_test_split()
# and split_map.
#
# This fixes the previous KeyError: 28765.
#
for i, group in enumerate(
    unique_groups
):

    # Correct zero-based lookup
    split = split_map[i]

    class_name = group["class"]

    source = group["source"]

    # Preserve source extension
    extension = source.suffix.lower()

    # Hash-based unique filename
    base_name = (
        f"{group['hash'][:16]}"
        f"{extension}"
    )

    destination = (
        OUTPUT /
        split /
        class_name /
        base_name
    )

    shutil.copy2(
        source,
        destination
    )

    manifest.append(
        {
            "split": split,
            "class": class_name,
            "source": str(source),
            "destination": str(destination),
            "sha256": group["hash"],
            "duplicate_group_size":
                group["duplicate_count"]
        }
    )

    if (i + 1) % 1000 == 0:

        print(
            f"Copied "
            f"{i + 1}/{len(unique_groups)}"
        )


# ============================================================
# VERIFY COPIED FILE COUNT
# ============================================================

print("\nVerifying copied files...")

copied_count = 0

for split in [
    "train",
    "val",
    "test"
]:

    split_dir = OUTPUT / split

    for path in split_dir.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            copied_count += 1


print(
    f"Expected files : {len(unique_groups)}"
)

print(
    f"Copied files   : {copied_count}"
)

if copied_count != len(unique_groups):

    raise RuntimeError(
        "\nCopied file count does not match "
        "unique image count."
    )

print(
    "File-count verification PASSED."
)


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_file = (
    METADATA /
    "rbc_final_manifest.csv"
)

with open(
    manifest_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "class",
            "source",
            "destination",
            "sha256",
            "duplicate_group_size"
        ]
    )

    writer.writeheader()

    writer.writerows(
        manifest
    )


# ============================================================
# SAVE CLASS MAP
# ============================================================

class_map = {
    str(index): class_name
    for index, class_name
    in enumerate(classes)
}

class_map_file = (
    METADATA /
    "rbc_classes.json"
)

with open(
    class_map_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_map,
        f,
        indent=4
    )


# ============================================================
# SAVE DATASET INFO
# ============================================================

info = {

    "source_images":
        len(records),

    "unique_images":
        len(unique_groups),

    "duplicate_copies_removed":
        len(records) - len(unique_groups),

    "classes":
        classes,

    "num_classes":
        len(classes),

    "train_images":
        len(train_idx),

    "validation_images":
        len(val_idx),

    "test_images":
        len(test_idx),

    "seed":
        SEED,

    "split":
        {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO
        }

}

info_file = (
    METADATA /
    "rbc_dataset_info.json"
)

with open(
    info_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        info,
        f,
        indent=4
    )


# ============================================================
# FINAL CLASS/SPLIT REPORT
# ============================================================

report = defaultdict(
    Counter
)

for item in manifest:

    report[
        item["class"]
    ][
        item["split"]
    ] += 1


report_file = (
    METADATA /
    "rbc_prepared_distribution.csv"
)

with open(
    report_file,
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
            "total"
        ]
    )

    for class_name in classes:

        train_count = report[
            class_name
        ]["train"]

        val_count = report[
            class_name
        ]["val"]

        test_count = report[
            class_name
        ]["test"]

        total = (
            train_count
            + val_count
            + test_count
        )

        writer.writerow(
            [
                class_name,
                train_count,
                val_count,
                test_count,
                total
            ]
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("RBC DATASET PREPARATION COMPLETE")
print("=" * 75)

print(
    f"\nOriginal images : {len(records)}"
)

print(
    f"Unique images   : {len(unique_groups)}"
)

print(
    f"Duplicates removed : "
    f"{len(records) - len(unique_groups)}"
)

print(
    f"\nTRAIN : {len(train_idx)}"
)

print(
    f"VAL   : {len(val_idx)}"
)

print(
    f"TEST  : {len(test_idx)}"
)

print(
    f"\nPrepared dataset:"
    f"\n{OUTPUT}"
)

print(
    f"\nManifest:"
    f"\n{manifest_file}"
)

print(
    f"\nClass mapping:"
    f"\n{class_map_file}"
)

print(
    f"\nDataset information:"
    f"\n{info_file}"
)

print(
    f"\nDistribution:"
    f"\n{report_file}"
)

print("\nOriginal dataset was NOT modified.")

print("\nReady for RBC morphology training.")