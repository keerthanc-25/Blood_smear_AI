import os
import csv
import random
import shutil
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\Blood_Smear_AI"

MANIFEST_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "annotation_manifest.csv"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_DIR,
    "data",
    "yolo_dataset"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# CHECK RATIOS
# ============================================================

if abs(
    TRAIN_RATIO +
    VAL_RATIO +
    TEST_RATIO - 1.0
) > 0.001:

    raise ValueError(
        "Train/Val/Test ratios must equal 1.0"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 5 - TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Check manifest
    # --------------------------------------------------------

    if not os.path.exists(MANIFEST_FILE):

        print("\nERROR:")
        print("Annotation manifest not found:")
        print(MANIFEST_FILE)

        return

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    print(
        "\nVerified annotated images:",
        f"{len(rows):,}"
    )

    # --------------------------------------------------------
    # Remove duplicate image paths
    # --------------------------------------------------------

    unique = {}

    for row in rows:

        image = os.path.normcase(
            os.path.normpath(
                row["image_file"]
            )
        )

        unique[image] = row

    rows = list(
        unique.values()
    )

    print(
        "Unique images:",
        f"{len(rows):,}"
    )

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    usable_rows = []

    missing = 0

    for row in rows:

        image_file = row["image_file"]
        label_file = row["label_file"]

        if (
            os.path.isfile(image_file)
            and
            os.path.isfile(label_file)
        ):

            usable_rows.append(row)

        else:

            missing += 1

    if missing > 0:

        print(
            "\nWARNING:",
            missing,
            "records have missing files."
        )

    rows = usable_rows

    print(
        "Usable image/label pairs:",
        f"{len(rows):,}"
    )

    if len(rows) == 0:

        print(
            "\nERROR: No usable images found."
        )

        return

    # --------------------------------------------------------
    # Randomize
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(rows)

    # --------------------------------------------------------
    # Calculate split sizes
    # --------------------------------------------------------

    total = len(rows)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    test_count = (
        total
        - train_count
        - val_count
    )

    train_rows = rows[
        :train_count
    ]

    val_rows = rows[
        train_count:
        train_count + val_count
    ]

    test_rows = rows[
        train_count + val_count:
    ]

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        os.makedirs(
            os.path.join(
                OUTPUT_ROOT,
                "images",
                split
            ),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(
                OUTPUT_ROOT,
                "labels",
                split
            ),
            exist_ok=True
        )

    # --------------------------------------------------------
    # Copy files
    # --------------------------------------------------------

    split_data = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows
    }

    split_class_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter()
    }

    split_object_counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    print("\nCopying files...\n")

    for split, split_rows in split_data.items():

        total_split = len(
            split_rows
        )

        for index, row in enumerate(
            split_rows,
            1
        ):

            source_image = row[
                "image_file"
            ]

            source_label = row[
                "label_file"
            ]

            # ------------------------------------------------
            # Unique filename
            # ------------------------------------------------

            image_name = os.path.basename(
                source_image
            )

            base_name = os.path.splitext(
                image_name
            )[0]

            extension = os.path.splitext(
                image_name
            )[1]

            # Add index to prevent collisions
            new_base = (
                f"{split}_{index:06d}_"
                f"{base_name}"
            )

            destination_image = os.path.join(
                OUTPUT_ROOT,
                "images",
                split,
                new_base + extension
            )

            destination_label = os.path.join(
                OUTPUT_ROOT,
                "labels",
                split,
                new_base + ".txt"
            )

            # ------------------------------------------------
            # Copy
            # ------------------------------------------------

            shutil.copy2(
                source_image,
                destination_image
            )

            shutil.copy2(
                source_label,
                destination_label
            )

            # ------------------------------------------------
            # Count objects
            # ------------------------------------------------

            try:

                with open(
                    source_label,
                    "r",
                    encoding="utf-8"
                ) as f:

                    label_lines = [
                        x.strip()
                        for x in f
                        if x.strip()
                    ]

                split_object_counts[
                    split
                ] += len(label_lines)

                for line in label_lines:

                    parts = line.split()

                    if len(parts) == 5:

                        class_id = int(
                            parts[0]
                        )

                        split_class_counts[
                            split
                        ][class_id] += 1

            except Exception:

                pass

            if (
                index % 500 == 0
                or
                index == total_split
            ):

                print(
                    f"{split.upper():5s}: "
                    f"{index:,}/{total_split:,}"
                )

    # --------------------------------------------------------
    # Save split manifests
    # --------------------------------------------------------

    for split, split_rows in split_data.items():

        manifest_path = os.path.join(
            OUTPUT_ROOT,
            f"{split}_manifest.csv"
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            fieldnames = [
                "image_file",
                "label_file",
                "original_image",
                "original_label"
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for index, row in enumerate(
                split_rows,
                1
            ):

                image_name = os.path.basename(
                    row["image_file"]
                )

                base_name = os.path.splitext(
                    image_name
                )[0]

                extension = os.path.splitext(
                    image_name
                )[1]

                new_base = (
                    f"{split}_{index:06d}_"
                    f"{base_name}"
                )

                writer.writerow({

                    "image_file":
                        os.path.join(
                            OUTPUT_ROOT,
                            "images",
                            split,
                            new_base + extension
                        ),

                    "label_file":
                        os.path.join(
                            OUTPUT_ROOT,
                            "labels",
                            split,
                            new_base + ".txt"
                        ),

                    "original_image":
                        row["image_file"],

                    "original_label":
                        row["label_file"]
                })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 5 COMPLETE")
    print("=" * 70)

    print("\nIMAGE COUNTS")

    print(
        f"Train      : {len(train_rows):,}"
    )

    print(
        f"Validation : {len(val_rows):,}"
    )

    print(
        f"Test       : {len(test_rows):,}"
    )

    print(
        f"Total      : {len(rows):,}"
    )

    print("\nPERCENTAGES")

    print(
        f"Train      : "
        f"{len(train_rows) / total * 100:.2f}%"
    )

    print(
        f"Validation : "
        f"{len(val_rows) / total * 100:.2f}%"
    )

    print(
        f"Test       : "
        f"{len(test_rows) / total * 100:.2f}%"
    )

    print("\nOBJECT COUNTS")

    for split in [
        "train",
        "val",
        "test"
    ]:

        print(
            f"\n{split.upper()}"
        )

        print(
            "Total objects:",
            f"{split_object_counts[split]:,}"
        )

        for class_id in range(4):

            name = {
                0: "RBC",
                1: "WBC",
                2: "Platelet",
                3: "Malaria_Parasite"
            }[class_id]

            print(
                f"  {name:18s}: "
                f"{split_class_counts[split][class_id]:,}"
            )

    # --------------------------------------------------------
    # Output structure
    # --------------------------------------------------------

    print("\nDATASET STRUCTURE")

    print(
        OUTPUT_ROOT
    )

    print("\n" + "=" * 70)
    print("NEXT STEP")
    print("=" * 70)

    print(
        "\nStep 6 will create data.yaml "
        "and perform the final dataset check."
    )


if __name__ == "__main__":
    main()