from pathlib import Path
import shutil

FINAL = Path(r"D:\Blood_Smear_AI\data\final_dataset")
VAHADANE = Path(r"D:\Blood_Smear_AI\data\final_dataset_vahadane")

final_images = FINAL / "images" / "train"
final_labels = FINAL / "labels" / "train"

vah_images = VAHADANE / "images" / "train"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}

print("=" * 70)
print("FAST VAHADANE MERGE")
print("=" * 70)

vah_files = sorted(
    [
        p for p in vah_images.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)

total = len(vah_files)

print(f"Vahadane images found: {total}")
print(f"Final training images before merge: {len(list(final_images.glob('*')))}")
print()

added = 0
missing_label = 0
already_exists = 0

for index, image in enumerate(vah_files, start=1):

    # --------------------------------------------------------
    # Find original label
    # --------------------------------------------------------

    source_label = final_labels / f"{image.stem}.txt"

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    print(
        f"[{index:3d}/{total}] "
        f"{image.name}",
        end=" "
    )

    if not source_label.exists():

        missing_label += 1

        print("-> MISSING LABEL")

        continue

    # --------------------------------------------------------
    # Destination names
    # --------------------------------------------------------

    destination_image = (
        final_images /
        f"vahadane_{image.name}"
    )

    destination_label = (
        final_labels /
        f"vahadane_{image.stem}.txt"
    )

    # --------------------------------------------------------
    # Avoid filename collision
    # --------------------------------------------------------

    counter = 1

    while destination_image.exists():

        destination_image = (
            final_images /
            f"vahadane_{counter}_{image.name}"
        )

        destination_label = (
            final_labels /
            f"vahadane_{counter}_{image.stem}.txt"
        )

        counter += 1

    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    shutil.copy2(
        image,
        destination_image
    )

    # --------------------------------------------------------
    # Copy label
    # --------------------------------------------------------

    shutil.copy2(
        source_label,
        destination_label
    )

    added += 1

    print("-> ADDED")


# ============================================================
# FINAL COUNTS
# ============================================================

final_image_count = len(
    [
        p for p in final_images.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)

final_label_count = len(
    list(final_labels.glob("*.txt"))
)


print()
print("=" * 70)
print("MERGE COMPLETE")
print("=" * 70)

print(f"Vahadane images processed : {total}")
print(f"Images added              : {added}")
print(f"Missing labels            : {missing_label}")
print(f"Already existing          : {already_exists}")
print()
print(f"Final training images     : {final_image_count}")
print(f"Final training labels     : {final_label_count}")
print("=" * 70)