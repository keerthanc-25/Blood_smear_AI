from pathlib import Path
from collections import Counter
from PIL import Image
import csv
import hashlib


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(
    r"D:\Blood_Smear_AI\data\wbc_morphology\raw"
)

METADATA = Path(
    r"D:\Blood_Smear_AI\data\wbc_morphology\metadata"
)

METADATA.mkdir(
    parents=True,
    exist_ok=True
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


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("WBC MORPHOLOGY DATASET AUDIT")
print("=" * 75)

print(
    f"\nDataset path:\n{DATASET}"
)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET}"
    )


splits = [
    "train",
    "val",
    "test"
]


for split in splits:

    split_path = DATASET / split

    if not split_path.exists():

        raise FileNotFoundError(
            f"Missing split:\n{split_path}"
        )


# ============================================================
# FIND CLASSES
# ============================================================

classes = sorted({

    folder.name

    for split in splits

    for folder in (
        DATASET / split
    ).iterdir()

    if folder.is_dir()
})


print("\n")
print("=" * 75)
print("CLASSES")
print("=" * 75)

print(
    f"\nNumber of classes: "
    f"{len(classes)}"
)

for index, class_name in enumerate(
    classes
):

    print(
        f"{index:2d} : {class_name}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

distribution = {}

total_images = 0


print(
    f"\n{'Class':<25}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
    f"{'Total':>10}"
)

print("-" * 70)


for class_name in classes:

    counts = {}

    for split in splits:

        class_dir = (
            DATASET /
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

        counts[split] = count

    total = sum(
        counts.values()
    )

    distribution[class_name] = counts

    total_images += total

    print(
        f"{class_name:<25}"
        f"{counts['train']:>10}"
        f"{counts['val']:>10}"
        f"{counts['test']:>10}"
        f"{total:>10}"
    )


# ============================================================
# SPLIT TOTALS
# ============================================================

train_total = sum(
    distribution[c]["train"]
    for c in classes
)

val_total = sum(
    distribution[c]["val"]
    for c in classes
)

test_total = sum(
    distribution[c]["test"]
    for c in classes
)


print("-" * 70)

print(
    f"{'TOTAL':<25}"
    f"{train_total:>10}"
    f"{val_total:>10}"
    f"{test_total:>10}"
    f"{total_images:>10}"
)


# ============================================================
# IMAGE VALIDATION
# ============================================================

print("\n")
print("=" * 75)
print("VALIDATING IMAGES")
print("=" * 75)

valid_images = 0
corrupt_images = []

image_records = []


all_images = []


for split in splits:

    for class_name in classes:

        class_dir = (
            DATASET /
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

                all_images.append(
                    (
                        file,
                        split,
                        class_name
                    )
                )


total_to_check = len(
    all_images
)


for index, (
    file,
    split,
    class_name
) in enumerate(
    all_images,
    start=1
):

    try:

        with Image.open(file) as image:

            image.verify()

        valid_images += 1

    except Exception as error:

        corrupt_images.append({
            "path": str(file),
            "split": split,
            "class": class_name,
            "error": str(error)
        })


    if (
        index % 1000 == 0
        or
        index == total_to_check
    ):

        print(
            f"Checked "
            f"{index}/{total_to_check}"
        )


print("\n")
print("IMAGE VALIDATION")
print("-" * 70)

print(
    f"Valid images  : "
    f"{valid_images}"
)

print(
    f"Corrupt images: "
    f"{len(corrupt_images)}"
)


# ============================================================
# SAVE CORRUPT REPORT
# ============================================================

corrupt_file = (
    METADATA /
    "corrupt_images.csv"
)

with open(
    corrupt_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "path",
            "split",
            "class",
            "error"
        ]
    )

    writer.writeheader()

    writer.writerows(
        corrupt_images
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

print("\n")
print("=" * 75)
print("CHECKING DUPLICATES")
print("=" * 75)

hash_map = {}

for index, (
    file,
    split,
    class_name
) in enumerate(
    all_images,
    start=1
):

    try:

        sha256 = hashlib.sha256()

        with open(
            file,
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

        hash_map.setdefault(
            digest,
            []
        ).append(
            (
                str(file),
                split,
                class_name
            )
        )

    except Exception:
        continue


duplicate_groups = {

    digest: records

    for digest, records
    in hash_map.items()

    if len(records) > 1
}


duplicate_files = sum(
    len(records) - 1
    for records
    in duplicate_groups.values()
)


print(
    f"Duplicate groups : "
    f"{len(duplicate_groups)}"
)

print(
    f"Duplicate files  : "
    f"{duplicate_files}"
)


# ============================================================
# CROSS-SPLIT DUPLICATES
# ============================================================

cross_split_groups = []

for digest, records in (
    duplicate_groups.items()
):

    split_names = {
        record[1]
        for record in records
    }

    if len(split_names) > 1:

        cross_split_groups.append(
            (
                digest,
                records
            )
        )


cross_split_files = sum(
    len(records) - 1
    for _, records
    in cross_split_groups
)


print(
    f"Cross-split duplicate groups:"
    f" {len(cross_split_groups)}"
)

print(
    f"Cross-split duplicate files:"
    f" {cross_split_files}"
)


# ============================================================
# SAVE DUPLICATE REPORT
# ============================================================

duplicate_file = (
    METADATA /
    "duplicates.txt"
)

with open(
    duplicate_file,
    "w",
    encoding="utf-8"
) as f:

    for digest, records in (
        duplicate_groups.items()
    ):

        f.write(
            f"\nHASH: {digest}\n"
        )

        for path, split, class_name in records:

            f.write(
                f"  {split} | "
                f"{class_name} | "
                f"{path}\n"
            )


# ============================================================
# SAVE CROSS-SPLIT REPORT
# ============================================================

cross_split_file = (
    METADATA /
    "cross_split_duplicates.txt"
)

with open(
    cross_split_file,
    "w",
    encoding="utf-8"
) as f:

    for digest, records in (
        cross_split_groups
    ):

        f.write(
            f"\nHASH: {digest}\n"
        )

        for path, split, class_name in records:

            f.write(
                f"  {split} | "
                f"{class_name} | "
                f"{path}\n"
            )


# ============================================================
# SAVE CLASS DISTRIBUTION
# ============================================================

distribution_file = (
    METADATA /
    "class_distribution.csv"
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

        writer.writerow([
            class_name,
            distribution[class_name]["train"],
            distribution[class_name]["val"],
            distribution[class_name]["test"],
            sum(
                distribution[class_name].values()
            )
        ])


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("AUDIT COMPLETE")
print("=" * 75)

print(
    f"\nTotal images : "
    f"{total_images}"
)

print(
    f"Classes      : "
    f"{len(classes)}"
)

print(
    f"Valid images : "
    f"{valid_images}"
)

print(
    f"Corrupt      : "
    f"{len(corrupt_images)}"
)

print(
    f"\nClass report:"
    f"\n{distribution_file}"
)

print(
    f"\nCorrupt report:"
    f"\n{corrupt_file}"
)

print(
    f"\nDuplicate report:"
    f"\n{duplicate_file}"
)

print(
    f"\nCross-split report:"
    f"\n{cross_split_file}"
)

print("\nOriginal dataset was NOT modified.")