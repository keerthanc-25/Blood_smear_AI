from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib
import csv

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    r"D:\Blood_Smear_AI\data\rbc_morphology\raw\RBC"
)

OUTPUT = Path(
    r"D:\Blood_Smear_AI\data\rbc_morphology\metadata"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
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
# HASH
# ============================================================

def get_hash(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# BASIC VALIDATION
# ============================================================

print("=" * 75)
print("RBC MORPHOLOGY SPLIT VALIDATION")
print("=" * 75)

print(f"\nDataset root:")
print(ROOT)

if not ROOT.exists():

    raise FileNotFoundError(
        f"Dataset not found: {ROOT}"
    )


# ============================================================
# COLLECT DATA
# ============================================================

all_records = []

for split in SPLITS:

    split_dir = ROOT / split

    if not split_dir.exists():

        print(
            f"\nWARNING: Missing split: {split}"
        )

        continue

    for class_dir in sorted(
        split_dir.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for image_path in class_dir.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                all_records.append(
                    {
                        "split": split,
                        "class": class_name,
                        "path": image_path
                    }
                )


# ============================================================
# SPLIT / CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

distribution = defaultdict(
    lambda: defaultdict(int)
)

for record in all_records:

    distribution[
        record["split"]
    ][
        record["class"]
    ] += 1


classes = sorted(
    {
        record["class"]
        for record in all_records
    }
)

print()

print(
    f"{'Class':<25}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
    f"{'Total':>10}"
)

print("-" * 65)

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

    print(
        f"{class_name:<25}"
        f"{train_count:>10}"
        f"{val_count:>10}"
        f"{test_count:>10}"
        f"{total:>10}"
    )


# ============================================================
# TOTALS
# ============================================================

print("-" * 65)

for split in SPLITS:

    count = sum(
        1
        for record in all_records
        if record["split"] == split
    )

    print(
        f"{split.upper():<25}"
        f"{count:>10}"
    )

print(
    f"\nTOTAL IMAGES: {len(all_records)}"
)


# ============================================================
# IMAGE VALIDATION
# ============================================================

print("\n")
print("=" * 75)
print("VALIDATING IMAGES")
print("=" * 75)

corrupt = []

for i, record in enumerate(
    all_records,
    1
):

    path = record["path"]

    try:

        with Image.open(path) as image:

            image.verify()

    except Exception as e:

        corrupt.append(
            {
                "split": record["split"],
                "class": record["class"],
                "path": str(path),
                "error": str(e)
            }
        )

    if i % 1000 == 0:

        print(
            f"Checked {i}/{len(all_records)}"
        )


print(
    f"\nCorrupt images: {len(corrupt)}"
)


# ============================================================
# HASH ALL IMAGES
# ============================================================

print("\n")
print("=" * 75)
print("CHECKING DUPLICATES")
print("=" * 75)

hash_map = defaultdict(list)

for i, record in enumerate(
    all_records,
    1
):

    path = record["path"]

    try:

        file_hash = get_hash(path)

        hash_map[file_hash].append(
            record
        )

    except Exception as e:

        print(
            f"Hash error: {path}"
        )
        print(e)

    if i % 1000 == 0:

        print(
            f"Hashed {i}/{len(all_records)}"
        )


duplicate_groups = {

    h: records

    for h, records
    in hash_map.items()

    if len(records) > 1
}


# ============================================================
# CROSS-SPLIT DUPLICATES
# ============================================================

cross_split_duplicates = []

same_split_duplicates = []

for file_hash, records in duplicate_groups.items():

    splits = {
        record["split"]
        for record in records
    }

    if len(splits) > 1:

        cross_split_duplicates.append(
            (file_hash, records)
        )

    else:

        same_split_duplicates.append(
            (file_hash, records)
        )


duplicate_file_count = sum(
    len(records) - 1
    for records in duplicate_groups.values()
)

cross_split_file_count = sum(
    len(records)
    for _, records
    in cross_split_duplicates
)


print("\n")
print("=" * 75)
print("DUPLICATE RESULTS")
print("=" * 75)

print(
    f"Duplicate groups          : "
    f"{len(duplicate_groups)}"
)

print(
    f"Duplicate files           : "
    f"{duplicate_file_count}"
)

print(
    f"Same-split duplicate groups: "
    f"{len(same_split_duplicates)}"
)

print(
    f"Cross-split duplicate groups: "
    f"{len(cross_split_duplicates)}"
)

print(
    f"Cross-split duplicate files : "
    f"{cross_split_file_count}"
)


# ============================================================
# SAVE CLASS DISTRIBUTION
# ============================================================

distribution_file = (
    OUTPUT /
    "rbc_split_distribution.csv"
)

with open(
    distribution_file,
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

        train_count = distribution[
            "train"
        ].get(class_name, 0)

        val_count = distribution[
            "val"
        ].get(class_name, 0)

        test_count = distribution[
            "test"
        ].get(class_name, 0)

        writer.writerow(
            [
                class_name,
                train_count,
                val_count,
                test_count,
                (
                    train_count
                    + val_count
                    + test_count
                )
            ]
        )


# ============================================================
# SAVE CROSS-SPLIT DUPLICATES
# ============================================================

duplicate_file = (
    OUTPUT /
    "cross_split_duplicates.txt"
)

with open(
    duplicate_file,
    "w",
    encoding="utf-8"
) as f:

    for index, (
        file_hash,
        records
    ) in enumerate(
        cross_split_duplicates,
        1
    ):

        f.write(
            f"\nGROUP {index}\n"
        )

        f.write(
            f"SHA256: {file_hash}\n"
        )

        for record in records:

            f.write(
                f"{record['split']} | "
                f"{record['class']} | "
                f"{record['path']}\n"
            )


# ============================================================
# SAVE CORRUPT FILES
# ============================================================

corrupt_file = (
    OUTPUT /
    "rbc_corrupt_images.csv"
)

with open(
    corrupt_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "split",
            "class",
            "path",
            "error"
        ]
    )

    for item in corrupt:

        writer.writerow(
            [
                item["split"],
                item["class"],
                item["path"],
                item["error"]
            ]
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("VALIDATION COMPLETE")
print("=" * 75)

print(
    f"\nDistribution report:"
    f"\n{distribution_file}"
)

print(
    f"\nCross-split duplicate report:"
    f"\n{duplicate_file}"
)

print(
    f"\nCorrupt image report:"
    f"\n{corrupt_file}"
)

print("\nIMPORTANT:")
print(
    "Original dataset was NOT modified."
)