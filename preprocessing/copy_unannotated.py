from pathlib import Path
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path(r"D:\blood smear images")
DEST_DIR = Path(r"D:\Blood_Smear_AI\data\unannotated")

MAX_IMAGES = 1000

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

# Annotation extensions commonly used
ANNOTATION_EXTENSIONS = {
    ".txt", ".xml", ".json"
}


# ============================================================
# CREATE DESTINATION
# ============================================================

DEST_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("BLOOD SMEAR - UNANNOTATED IMAGE COPIER")
print("=" * 60)

print(f"\nSource : {SOURCE_DIR}")
print(f"Target : {DEST_DIR}")
print(f"Maximum images : {MAX_IMAGES}\n")


# ============================================================
# CHECK SOURCE
# ============================================================

if not SOURCE_DIR.exists():
    print(f"ERROR: Source folder does not exist:")
    print(SOURCE_DIR)
    raise SystemExit(1)


# ============================================================
# GET ALREADY COPIED FILES
# ============================================================

already_copied = {
    file.name.lower()
    for file in DEST_DIR.iterdir()
    if file.is_file()
}

print(f"Already present in destination: {len(already_copied)}")


# ============================================================
# FIND IMAGES
# ============================================================

print("\nScanning source folder...")

all_images = []

for file in SOURCE_DIR.rglob("*"):
    if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
        all_images.append(file)

print(f"Total image files found: {len(all_images)}")


# ============================================================
# DETERMINE WHETHER IMAGE IS ANNOTATED
# ============================================================

unannotated = []

for image in all_images:

    # --------------------------------------------------------
    # 1. Check annotation files beside the image
    # --------------------------------------------------------

    stem = image.stem
    parent = image.parent

    has_annotation = False

    for ext in ANNOTATION_EXTENSIONS:
        annotation_file = parent / f"{stem}{ext}"

        if annotation_file.exists():
            has_annotation = True
            break

    if has_annotation:
        continue

    # --------------------------------------------------------
    # 2. Check YOLO-style "labels" folder
    # --------------------------------------------------------

    # Example:
    # dataset/images/image1.jpg
    # dataset/labels/image1.txt

    labels_dir = parent.parent / "labels"

    if labels_dir.exists():
        yolo_label = labels_dir / f"{stem}.txt"

        if yolo_label.exists():
            has_annotation = True

    if has_annotation:
        continue

    # --------------------------------------------------------
    # 3. Check if already copied
    # --------------------------------------------------------

    if image.name.lower() in already_copied:
        continue

    unannotated.append(image)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(f"\nUnannotated candidates found: {len(unannotated)}")

if len(unannotated) == 0:
    print("\nNo unannotated images were found.")
    raise SystemExit(0)


# ============================================================
# COPY FIRST N IMAGES
# ============================================================

images_to_copy = unannotated[:MAX_IMAGES]

print(f"Images selected for copying: {len(images_to_copy)}")
print("\nCopying images...\n")


copied = 0
failed = 0

for index, image in enumerate(images_to_copy, start=1):

    destination = DEST_DIR / image.name

    # --------------------------------------------------------
    # Avoid filename collision
    # --------------------------------------------------------

    if destination.exists():

        counter = 1

        while destination.exists():
            destination = DEST_DIR / f"{image.stem}_{counter}{image.suffix}"
            counter += 1

    try:

        shutil.copy2(image, destination)

        copied += 1

        print(
            f"[{index}/{len(images_to_copy)}] "
            f"Copied: {image.name}"
        )

    except Exception as e:

        failed += 1

        print(
            f"[ERROR] {image.name} -> {e}"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("COPY COMPLETE")
print("=" * 60)

print(f"Images copied successfully : {copied}")
print(f"Failed                     : {failed}")

print(f"\nUnannotated folder:")
print(DEST_DIR)

print("\nNext step:")
print("Run the automatic YOLO pseudo-labeling script.")
print("=" * 60)