import os
import csv
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

SOURCE = Path(r"D:\blood smear images")

ORIGINAL = Path(r"D:\Blood_Smear_AI\data\yolo_dataset")

PSEUDO_GROUPS = [
    Path(r"D:\Blood_Smear_AI\data\pseudo_labeled_new\rbc"),
    Path(r"D:\Blood_Smear_AI\data\pseudo_labeled_new\wbc_platelet"),
    Path(r"D:\Blood_Smear_AI\data\rbc_second_pass"),
    Path(r"D:\Blood_Smear_AI\data\rbc_selected"),
]

OUTPUT = Path(r"D:\Blood_Smear_AI\data\dataset_inventory_fixed")

UNCLASSIFIED_DIR = Path(
    r"D:\Blood_Smear_AI\data\remaining_unclassified"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Original YOLO class IDs
RBC = 0
WBC = 1
PLATELET = 2
MALARIA = 3


# ============================================================
# HASH
# ============================================================

def file_hash(path):
    h = hashlib.sha256()

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)

        return h.hexdigest()

    except Exception:
        return None


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):
    if not folder.exists():
        return []

    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


# ============================================================
# ORIGINAL IMAGE HASHES
# ============================================================

print("=" * 70)
print("FIXED MASTER DATASET INVENTORY")
print("=" * 70)

print("\nScanning source...")

source_images = get_images(SOURCE)

print(f"Source images found: {len(source_images)}")


print("\nHashing SOURCE images...")

source_hashes = {}
invalid = 0

for i, img in enumerate(source_images, 1):

    h = file_hash(img)

    if h is None:
        invalid += 1
        continue

    source_hashes[h] = img

    if i % 10000 == 0:
        print(f"Hashed {i}/{len(source_images)}")


# ============================================================
# ORIGINAL DATASET HASHES
# ============================================================

print("\nScanning ORIGINAL annotated dataset...")

original_hashes = {}

for img in get_images(ORIGINAL):

    h = file_hash(img)

    if h is not None:
        original_hashes[h] = img

print(f"Original unique images: {len(original_hashes)}")


# ============================================================
# DETERMINE ORIGINAL CLASSES
# ============================================================

print("\nReading original labels...")

original_category = {}

for h, img in original_hashes.items():

    label = img.parent.parent.parent / "labels" / img.parent.name / f"{img.stem}.txt"

    # Alternative search if structure differs
    if not label.exists():

        candidates = list(
            ORIGINAL.rglob(f"{img.stem}.txt")
        )

        if candidates:
            label = candidates[0]

    classes = set()

    if label.exists():

        try:
            with open(label, "r", encoding="utf-8") as f:

                for line in f:

                    parts = line.strip().split()

                    if parts:
                        classes.add(int(parts[0]))

        except Exception:
            pass

    if MALARIA in classes:
        original_category[h] = "original_annotated_malaria"

    elif classes & {RBC, WBC, PLATELET}:
        original_category[h] = "original_annotated_3class"

    else:
        original_category[h] = "original_annotated_other"


# ============================================================
# PSEUDO HASH UNION
# ============================================================

print("\nHashing pseudo-labelled datasets...")

pseudo_hashes = set()

pseudo_group_counts = defaultdict(set)

for group in PSEUDO_GROUPS:

    images = get_images(group)

    print(f"{group.name}: {len(images)} images")

    for img in images:

        h = file_hash(img)

        if h is None:
            continue

        pseudo_group_counts[group.name].add(h)
        pseudo_hashes.add(h)


print(f"\nUnique pseudo-labelled images: {len(pseudo_hashes)}")


# ============================================================
# MUTUALLY EXCLUSIVE CLASSIFICATION
# ============================================================

print("\nCreating mutually exclusive inventory...")

categories = defaultdict(list)

for h, img in source_hashes.items():

    # Priority 1: original annotations
    if h in original_category:

        categories[original_category[h]].append(img)
        continue

    # Priority 2: pseudo annotations
    if h in pseudo_hashes:

        categories["pseudo_annotated"].append(img)
        continue

    # Priority 3: genuinely unclassified
    categories["remaining_unclassified"].append(img)


# ============================================================
# COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED INVENTORY")
print("=" * 70)

for name in [
    "original_annotated_3class",
    "original_annotated_malaria",
    "original_annotated_other",
    "pseudo_annotated",
    "remaining_unclassified",
]:

    print(f"{name:<32}: {len(categories[name])}")


total = sum(len(v) for v in categories.values())

print("-" * 70)
print(f"{'SOURCE HASHED IMAGES':<32}: {len(source_hashes)}")
print(f"{'CATEGORY TOTAL':<32}: {total}")
print(f"{'DIFFERENCE':<32}: {len(source_hashes) - total}")


# ============================================================
# OUTPUT
# ============================================================

OUTPUT.mkdir(parents=True, exist_ok=True)

UNCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

# CSV
csv_path = OUTPUT / "source_image_inventory.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "sha256",
        "source_path",
        "category"
    ])

    for h, img in source_hashes.items():

        if h in original_category:
            category = original_category[h]

        elif h in pseudo_hashes:
            category = "pseudo_annotated"

        else:
            category = "remaining_unclassified"

        writer.writerow([
            h,
            str(img),
            category
        ])


# ============================================================
# COPY UNCLASSIFIED
# ============================================================

print("\nExtracting truly unclassified images...")

unclassified_csv = OUTPUT / "remaining_unclassified.csv"

with open(unclassified_csv, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "sha256",
        "source_path"
    ])

    for i, img in enumerate(categories["remaining_unclassified"], 1):

        h = file_hash(img)

        # Preserve original extension
        destination = UNCLASSIFIED_DIR / f"unclassified_{i:06d}{img.suffix.lower()}"

        try:
            shutil.copy2(img, destination)

            writer.writerow([
                h,
                str(img)
            ])

        except Exception as e:
            print(f"Copy failed: {img} -> {e}")

        if i % 5000 == 0:
            print(
                f"Copied {i}/"
                f"{len(categories['remaining_unclassified'])}"
            )


# ============================================================
# SUMMARY
# ============================================================

summary_path = OUTPUT / "inventory_summary.csv"

with open(summary_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "category",
        "count"
    ])

    for name in [
        "original_annotated_3class",
        "original_annotated_malaria",
        "original_annotated_other",
        "pseudo_annotated",
        "remaining_unclassified",
    ]:

        writer.writerow([
            name,
            len(categories[name])
        ])


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"\nInventory:")
print(OUTPUT)

print("\nUnclassified images:")
print(UNCLASSIFIED_DIR)

print("\nNo source datasets were deleted.")