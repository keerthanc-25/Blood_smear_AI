import os
import yaml

PROJECT_DIR = r"D:\Blood_Smear_AI"

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "yolo_dataset"
)

YAML_FILE = os.path.join(
    DATASET_DIR,
    "data.yaml"
)


CLASSES = [
    "RBC",
    "WBC",
    "Platelet",
    "Malaria_Parasite"
]


def count_files(folder, extension):

    if not os.path.exists(folder):
        return 0

    return sum(
        1
        for file in os.listdir(folder)
        if file.lower().endswith(extension)
    )


def main():

    print("=" * 70)
    print("STEP 6 - CREATE YOLO DATASET CONFIGURATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check dataset folders
    # --------------------------------------------------------

    required_folders = [
        os.path.join(
            DATASET_DIR,
            "images",
            "train"
        ),
        os.path.join(
            DATASET_DIR,
            "images",
            "val"
        ),
        os.path.join(
            DATASET_DIR,
            "images",
            "test"
        ),
        os.path.join(
            DATASET_DIR,
            "labels",
            "train"
        ),
        os.path.join(
            DATASET_DIR,
            "labels",
            "val"
        ),
        os.path.join(
            DATASET_DIR,
            "labels",
            "test"
        )
    ]

    missing = []

    for folder in required_folders:

        if not os.path.isdir(folder):
            missing.append(folder)

    if missing:

        print("\nERROR: Missing folders:")

        for folder in missing:
            print(folder)

        return

    # --------------------------------------------------------
    # Count images and labels
    # --------------------------------------------------------

    train_images = count_files(
        os.path.join(
            DATASET_DIR,
            "images",
            "train"
        ),
        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    )

    val_images = count_files(
        os.path.join(
            DATASET_DIR,
            "images",
            "val"
        ),
        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    )

    test_images = count_files(
        os.path.join(
            DATASET_DIR,
            "images",
            "test"
        ),
        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    )

    train_labels = count_files(
        os.path.join(
            DATASET_DIR,
            "labels",
            "train"
        ),
        ".txt"
    )

    val_labels = count_files(
        os.path.join(
            DATASET_DIR,
            "labels",
            "val"
        ),
        ".txt"
    )

    test_labels = count_files(
        os.path.join(
            DATASET_DIR,
            "labels",
            "test"
        ),
        ".txt"
    )

    # --------------------------------------------------------
    # Print counts
    # --------------------------------------------------------

    print("\nDATASET COUNTS")
    print("-" * 70)

    print(
        f"Train images       : {train_images:,}"
    )

    print(
        f"Train labels       : {train_labels:,}"
    )

    print(
        f"Validation images  : {val_images:,}"
    )

    print(
        f"Validation labels  : {val_labels:,}"
    )

    print(
        f"Test images        : {test_images:,}"
    )

    print(
        f"Test labels        : {test_labels:,}"
    )

    # --------------------------------------------------------
    # Check image/label counts
    # --------------------------------------------------------

    print("\nIMAGE / LABEL CHECK")

    if train_images == train_labels:
        print("Train      : OK")
    else:
        print("Train      : WARNING")

    if val_images == val_labels:
        print("Validation : OK")
    else:
        print("Validation : WARNING")

    if test_images == test_labels:
        print("Test       : OK")
    else:
        print("Test       : WARNING")

    # --------------------------------------------------------
    # Create data.yaml
    # --------------------------------------------------------

    yaml_data = {

        "path": DATASET_DIR,

        "train": "images/train",

        "val": "images/val",

        "test": "images/test",

        "nc": len(CLASSES),

        "names": CLASSES
    }

    with open(
        YAML_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.dump(
            yaml_data,
            f,
            sort_keys=False
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\nDATA.YAML CREATED")
    print("-" * 70)

    print(YAML_FILE)

    print("\nCLASSES")

    for index, name in enumerate(CLASSES):

        print(
            f"{index} = {name}"
        )

    print("\n" + "=" * 70)
    print("STEP 6 COMPLETE")
    print("=" * 70)

    print("\nYour YOLO dataset is now configured.")

    print(
        "\nNext:"
        "\nSTEP 7 - FINAL DATASET SANITY CHECK"
    )


if __name__ == "__main__":
    main()