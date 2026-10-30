from pathlib import Path
from collections import defaultdict
import hashlib
import shutil
import csv


# ============================================================
# CONFIGURATION
# ============================================================

RAW = Path(
    r"D:\Blood_Smear_AI\data\wbc_morphology\raw"
)

PREPARED = Path(
    r"D:\Blood_Smear_AI\data\wbc_morphology\prepared"
)

METADATA = Path(
    r"D:\Blood_Smear_AI\data\wbc_morphology\metadata"
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
print("WBC MORPHOLOGY DATASET PREPARATION")
print("=" * 75)

print(
    f"\nSource:\n{RAW}"
)

print(
    f"\nOutput:\n{PREPARED}"
)


# ============================================================
# CHECK SOURCE
# ============================================================

if not RAW.exists():

    raise FileNotFoundError(
        f"Raw dataset not found:\n{RAW}"
    )


# ============================================================
# DISCOVER CLASSES
# ============================================================

classes = sorted({

    folder.name

    for split in SPLITS

    for folder in (
        RAW / split
    ).iterdir()

    if folder.is_dir()
})


print(
    f"\nClasses found: {len(classes)}"
)

for index, class_name in enumerate(
    classes
):

    print(
        f"{index:2d} : {class_name}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in SPLITS:

    for class_name in classes:

        (
            PREPARED /
            split /
            class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COLLECT ALL FILES
# ============================================================

records = []

for split in SPLITS:

    for class_name in classes:

        class_dir = (
            RAW /
            split /
            class_name
        )

        for file in class_dir.iterdir():

            if (
                file.is_file()
                and
                file.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                records.append({
                    "path": file,
                    "split": split,
                    "class": class_name
                })


print(
    f"\nTotal source images: "
    f"{len(records)}"
)


# ============================================================
# HASH IMAGES
# ============================================================

print("\nCalculating image hashes...")

hash_groups = defaultdict(list)


for index, record in enumerate(
    records,
    start=1
):

    sha256 = hashlib.sha256()

    with open(
        record["path"],
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    digest = sha256.hexdigest()

    hash_groups[digest].append(
        record
    )

    if (
        index % 1000 == 0
        or
        index == len(records)
    ):

        print(
            f"Hashed "
            f"{index}/{len(records)}"
        )


# ============================================================
# IDENTIFY CROSS-SPLIT DUPLICATES
# ============================================================

cross_split_duplicates = []

for digest, group in hash_groups.items():

    split_names = {
        record["split"]
        for record in group
    }

    if len(split_names) > 1:

        cross_split_duplicates.append(
            {
                "hash": digest,
                "records": group
            }
        )


print("\n")
print("=" * 75)
print("CROSS-SPLIT DUPLICATE ANALYSIS")
print("=" * 75)

print(
    f"\nCross-split duplicate groups: "
    f"{len(cross_split_duplicates)}"
)


# ============================================================
# DETERMINE FILES TO EXCLUDE
# ============================================================

exclude_paths = set()

for duplicate in cross_split_duplicates:

    group = duplicate["records"]

    print("\nDuplicate group:")

    for record in group:

        print(
            f"  {record['split']:5s} | "
            f"{record['class']:25s} | "
            f"{record['path']}"
        )

    # --------------------------------------------------------
    # Priority:
    #
    # Keep VAL/TEST
    # Remove duplicate from TRAIN
    #
    # If there is no train copy, keep the first occurrence.
    # --------------------------------------------------------

    train_records = [
        record
        for record in group
        if record["split"] == "train"
    ]

    if train_records:

        for record in train_records:

            exclude_paths.add(
                str(
                    record["path"]
                )
            )

    else:

        # Extremely unlikely for this dataset,
        # but keeps the function safe.

        for record in group[1:]:

            exclude_paths.add(
                str(
                    record["path"]
                )
            )


print(
    f"\nImages excluded from prepared dataset: "
    f"{len(exclude_paths)}"
)


# ============================================================
# COPY DATASET
# ============================================================

print("\n")
print("=" * 75)
print("COPYING DATASET")
print("=" * 75)


copied = 0
skipped = 0


for index, record in enumerate(
    records,
    start=1
):

    source = record["path"]

    if str(source) in exclude_paths:

        skipped += 1

        continue

    destination_dir = (
        PREPARED /
        record["split"] /
        record["class"]
    )

    destination = (
        destination_dir /
        source.name
    )

    # --------------------------------------------------------
    # Prevent filename collision
    # --------------------------------------------------------

    if destination.exists():

        stem = source.stem
        suffix = source.suffix

        counter = 1

        while destination.exists():

            destination = (
                destination_dir /
                f"{stem}_copy{counter}"
                f"{suffix}"
            )

            counter += 1

    shutil.copy2(
        source,
        destination
    )

    copied += 1

    if (
        index % 1000 == 0
        or
        index == len(records)
    ):

        print(
            f"Processed "
            f"{index}/{len(records)}"
        )


# ============================================================
# VERIFY COUNTS
# ============================================================

print("\n")
print("=" * 75)
print("FILE-COUNT VERIFICATION")
print("=" * 75)


final_counts = {}

for split in SPLITS:

    split_total = 0

    for class_name in classes:

        class_dir = (
            PREPARED /
            split /
            class_name
        )

        count = sum(

            1

            for file in class_dir.iterdir()

            if (
                file.is_file()
                and
                file.suffix.lower()
                in IMAGE_EXTENSIONS
            )

        )

        final_counts[
            (
                split,
                class_name
            )
        ] = count

        split_total += count

    print(
        f"{split:<8}: "
        f"{split_total}"
    )


# ============================================================
# SAVE DISTRIBUTION
# ============================================================

distribution_file = (
    METADATA /
    "wbc_prepared_distribution.csv"
)

with open(
    distribution_file,
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
        "total"
    ])

    for class_name in classes:

        train_count = final_counts[
            ("train", class_name)
        ]

        val_count = final_counts[
            ("val", class_name)
        ]

        test_count = final_counts[
            ("test", class_name)
        ]

        writer.writerow([
            class_name,
            train_count,
            val_count,
            test_count,
            (
                train_count +
                val_count +
                test_count
            )
        ])


# ============================================================
# SAVE EXCLUSION REPORT
# ============================================================

excluded_file = (
    METADATA /
    "cross_split_excluded.txt"
)

with open(
    excluded_file,
    "w",
    encoding="utf-8"
) as f:

    for path in sorted(
        exclude_paths
    ):

        f.write(
            path + "\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

train_total = sum(
    final_counts[
        ("train", class_name)
    ]
    for class_name in classes
)

val_total = sum(
    final_counts[
        ("val", class_name)
    ]
    for class_name in classes
)

test_total = sum(
    final_counts[
        ("test", class_name)
    ]
    for class_name in classes
)


print("\n")
print("=" * 75)
print("WBC PREPARATION COMPLETE")
print("=" * 75)

print(
    f"\nOriginal images : "
    f"{len(records)}"
)

print(
    f"Excluded        : "
    f"{len(exclude_paths)}"
)

print(
    f"Prepared images : "
    f"{copied}"
)

print(
    f"\nTRAIN : "
    f"{train_total}"
)

print(
    f"VAL   : "
    f"{val_total}"
)

print(
    f"TEST  : "
    f"{test_total}"
)

print(
    f"\nPrepared dataset:"
    f"\n{PREPARED}"
)

print(
    f"\nDistribution:"
    f"\n{distribution_file}"
)

print(
    f"\nExcluded duplicates:"
    f"\n{excluded_file}"
)

print(
    "\nOriginal dataset was NOT modified."
)

print(
    "\nReady for WBC class-balance analysis."
)