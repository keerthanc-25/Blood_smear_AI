import os
from collections import Counter

PROJECT_DIR = r"D:\Blood_Smear_AI"

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "yolo_dataset"
)

CLASSES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet",
    3: "Malaria_Parasite"
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


def get_files(folder, extensions):

    if not os.path.isdir(folder):
        return []

    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower()
        in extensions
    ]


def main():

    print("=" * 70)
    print("STEP 7 - FINAL DATASET SANITY CHECK")
    print("=" * 70)

    total_images = 0
    total_labels = 0
    total_objects = 0

    class_counts = Counter()

    all_ok = True

    for split in ["train", "val", "test"]:

        image_dir = os.path.join(
            DATASET_DIR,
            "images",
            split
        )

        label_dir = os.path.join(
            DATASET_DIR,
            "labels",
            split
        )

        images = get_files(
            image_dir,
            IMAGE_EXTENSIONS
        )

        labels = get_files(
            label_dir,
            {".txt"}
        )

        image_names = {
            os.path.splitext(
                os.path.basename(x)
            )[0]
            for x in images
        }

        label_names = {
            os.path.splitext(
                os.path.basename(x)
            )[0]
            for x in labels
        }

        missing_labels = (
            image_names - label_names
        )

        missing_images = (
            label_names - image_names
        )

        split_objects = 0

        for label in labels:

            try:

                with open(
                    label,
                    "r",
                    encoding="utf-8"
                ) as f:

                    for line in f:

                        parts = line.strip().split()

                        if len(parts) != 5:
                            all_ok = False
                            continue

                        try:

                            class_id = int(parts[0])

                            xc = float(parts[1])
                            yc = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])

                        except ValueError:

                            all_ok = False
                            continue

                        if class_id not in CLASSES:
                            all_ok = False
                            continue

                        if not (
                            0 <= xc <= 1
                            and 0 <= yc <= 1
                            and 0 < w <= 1
                            and 0 < h <= 1
                        ):

                            all_ok = False
                            continue

                        class_counts[class_id] += 1
                        split_objects += 1

            except Exception:

                all_ok = False

        if missing_labels:
            all_ok = False

        if missing_images:
            all_ok = False

        print(f"\n{split.upper()}")
        print("-" * 40)

        print(
            f"Images       : {len(images):,}"
        )

        print(
            f"Labels       : {len(labels):,}"
        )

        print(
            f"Objects      : {split_objects:,}"
        )

        print(
            f"Missing labels: {len(missing_labels):,}"
        )

        print(
            f"Missing images: {len(missing_images):,}"
        )

        total_images += len(images)
        total_labels += len(labels)
        total_objects += split_objects

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal images  : {total_images:,}"
    )

    print(
        f"Total labels  : {total_labels:,}"
    )

    print(
        f"Total objects : {total_objects:,}"
    )

    print("\nCLASS DISTRIBUTION")

    for class_id in range(4):

        print(
            f"{class_id} = "
            f"{CLASSES[class_id]:18s} : "
            f"{class_counts[class_id]:,}"
        )

    print("\nFINAL CHECK")

    if (
        total_images == total_labels
        and
        all_ok
        and
        total_images == 5310
    ):

        print("\n✓ IMAGE/LABEL MATCH      : PASS")
        print("✓ YOLO LABEL FORMAT      : PASS")
        print("✓ CLASS IDs              : PASS")
        print("✓ BOUNDING BOXES         : PASS")
        print("✓ TRAIN/VAL/TEST DATA    : PASS")

        print("\n" + "=" * 70)
        print("OBJECTIVE 1 COMPLETE")
        print("=" * 70)

        print(
            "\nThe verified annotated dataset "
            "is ready for AI model training."
        )

    else:

        print("\n✗ DATASET CHECK FAILED")

        print(
            "Review the problems above before training."
        )


if __name__ == "__main__":
    main()