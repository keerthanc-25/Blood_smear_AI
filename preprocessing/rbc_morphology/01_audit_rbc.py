from pathlib import Path
from PIL import Image
import hashlib
import csv

# ============================================================
# CONFIG
# ============================================================

DATASET = Path(r"D:\Blood_Smear_AI\data\rbc_morphology\raw")

OUTPUT = Path(
    r"D:\Blood_Smear_AI\data\rbc_morphology\metadata"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

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
# HELPERS
# ============================================================

def file_hash(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# FIND IMAGES
# ============================================================

images = []

for path in DATASET.rglob("*"):

    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        images.append(path)


print("=" * 70)
print("RBC MORPHOLOGY DATASET AUDIT")
print("=" * 70)

print(f"\nDataset path : {DATASET}")
print(f"Total images : {len(images)}")


# ============================================================
# CLASS DETECTION
# ============================================================

class_counts = {}

for img in images:

    relative = img.relative_to(DATASET)

    parts = relative.parts

    if len(parts) >= 2:
        class_name = parts[-2]
    else:
        class_name = "UNKNOWN"

    class_counts[class_name] = class_counts.get(class_name, 0) + 1


print("\nCLASS DISTRIBUTION")
print("-" * 70)

for class_name, count in sorted(
    class_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{class_name:<35} {count:>8}")


# ============================================================
# IMAGE VALIDATION
# ============================================================

print("\nVALIDATING IMAGES...")

corrupt = []
dimensions = {}

for i, img in enumerate(images, 1):

    try:

        with Image.open(img) as im:

            im.verify()

        with Image.open(img) as im:

            size = im.size

        dimensions[size] = dimensions.get(size, 0) + 1

    except Exception as e:

        corrupt.append(
            (str(img), str(e))
        )

    if i % 500 == 0:
        print(f"Checked {i}/{len(images)}")


print("\nIMAGE VALIDATION")
print("-" * 70)

print(f"Valid images   : {len(images) - len(corrupt)}")
print(f"Corrupt images : {len(corrupt)}")


# ============================================================
# DUPLICATE DETECTION
# ============================================================

print("\nCHECKING DUPLICATES...")

hash_map = {}

for i, img in enumerate(images, 1):

    try:

        h = file_hash(img)

        hash_map.setdefault(h, []).append(img)

    except Exception:
        pass

duplicates = {
    h: paths
    for h, paths in hash_map.items()
    if len(paths) > 1
}

duplicate_images = sum(
    len(paths) - 1
    for paths in duplicates.values()
)


print("\nDUPLICATE SUMMARY")
print("-" * 70)

print(f"Duplicate groups : {len(duplicates)}")
print(f"Duplicate files  : {duplicate_images}")


# ============================================================
# FORMAT DISTRIBUTION
# ============================================================

formats = {}

for img in images:

    ext = img.suffix.lower()

    formats[ext] = formats.get(ext, 0) + 1


print("\nIMAGE FORMATS")
print("-" * 70)

for ext, count in sorted(formats.items()):

    print(f"{ext:<10} {count:>8}")


# ============================================================
# DIMENSIONS
# ============================================================

print("\nTOP IMAGE DIMENSIONS")
print("-" * 70)

for size, count in sorted(
    dimensions.items(),
    key=lambda x: x[1],
    reverse=True
)[:15]:

    print(
        f"{str(size):<20} {count:>8}"
    )


# ============================================================
# SAVE CORRUPT FILES
# ============================================================

corrupt_file = OUTPUT / "corrupt_images.txt"

with open(
    corrupt_file,
    "w",
    encoding="utf-8"
) as f:

    for path, error in corrupt:

        f.write(
            f"{path}\t{error}\n"
        )


# ============================================================
# SAVE CLASS COUNTS
# ============================================================

class_file = OUTPUT / "class_distribution.csv"

with open(
    class_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["class", "image_count"]
    )

    for class_name, count in sorted(class_counts.items()):

        writer.writerow(
            [class_name, count]
        )


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(f"\nClass report : {class_file}")
print(f"Corrupt list : {corrupt_file}")