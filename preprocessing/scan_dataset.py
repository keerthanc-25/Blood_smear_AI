import os
import csv
from collections import Counter, defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\Blood_Smear_AI"

# Original dataset - DO NOT COPY
DATASET_ROOT = r"D:\blood smear images"

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "dataset_inventory.csv"
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

ANNOTATION_EXTENSIONS = {
    ".txt": "TXT",
    ".xml": "Pascal_VOC",
    ".json": "JSON",
    ".csv": "CSV",
    ".mat": "MATLAB",
    ".pts": "Point",
    ".poly": "Polygon"
}

ANNOTATION_FOLDER_NAMES = {
    "label",
    "labels",
    "annotation",
    "annotations",
    "annot",
    "groundtruth",
    "ground_truth",
    "gt",
    "gt_updated",
    "point set",
    "polygon set",
    "points",
    "polygons"
}


# ============================================================
# HELPERS
# ============================================================

def get_extension(filename):
    return os.path.splitext(filename)[1].lower()


def is_image(filename):
    return get_extension(filename) in IMAGE_EXTENSIONS


def is_annotation(filename):
    return get_extension(filename) in ANNOTATION_EXTENSIONS


def get_stem(filename):
    return os.path.splitext(filename)[0].lower()


def get_dataset_name(path):
    """
    First folder below the original dataset root.
    """

    relative = os.path.relpath(
        path,
        DATASET_ROOT
    )

    parts = relative.split(os.sep)

    if len(parts) > 1:
        return parts[0]

    return "ROOT"


def get_split(path):

    parts = [
        x.lower()
        for x in path.split(os.sep)
    ]

    for part in parts:

        if part in ("train", "training"):
            return "train"

        if part in ("val", "valid", "validation"):
            return "val"

        if part in ("test", "testing"):
            return "test"

    return ""


def get_annotation_type(files):

    types = []

    for file in files:

        annotation_type = ANNOTATION_EXTENSIONS.get(
            get_extension(file)
        )

        if annotation_type and annotation_type not in types:
            types.append(annotation_type)

    return ";".join(types)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 1 - FAST BLOOD SMEAR DATASET SCAN")
    print("=" * 75)

    print("\nProject:")
    print(PROJECT_DIR)

    print("\nOriginal dataset:")
    print(DATASET_ROOT)

    if not os.path.exists(DATASET_ROOT):

        print("\nERROR:")
        print("Dataset folder does not exist:")
        print(DATASET_ROOT)

        return

    print("\nScanning all files once...")
    print("Original dataset will NOT be modified.")
    print("Please wait...\n")

    # ========================================================
    # PASS 1
    # Scan everything ONCE
    # ========================================================

    image_files = []

    # Key:
    # (dataset_name, filename_stem)
    #
    # Value:
    # annotation files
    #
    annotation_index = defaultdict(list)

    image_extensions = Counter()
    annotation_extensions = Counter()
    dataset_counts = Counter()
    split_counts = Counter()

    total_files = 0

    for root, _, files in os.walk(DATASET_ROOT):

        dataset = get_dataset_name(root)

        for filename in files:

            total_files += 1

            full_path = os.path.join(
                root,
                filename
            )

            ext = get_extension(filename)
            stem = get_stem(filename)

            # ----------------------------------------------
            # Image
            # ----------------------------------------------

            if ext in IMAGE_EXTENSIONS:

                image_files.append(full_path)

                image_extensions[ext] += 1
                dataset_counts[dataset] += 1

                split = get_split(full_path)

                if split:
                    split_counts[split] += 1

            # ----------------------------------------------
            # Annotation
            # ----------------------------------------------

            elif ext in ANNOTATION_EXTENSIONS:

                annotation_extensions[ext] += 1

                annotation_index[
                    (dataset, stem)
                ].append(full_path)

    image_files.sort()

    # ========================================================
    # IMAGE SUMMARY
    # ========================================================

    print("=" * 75)
    print("IMAGE SUMMARY")
    print("=" * 75)

    print(
        "\nTotal images found:",
        len(image_files)
    )

    print("\nImage extensions:")

    for ext, count in image_extensions.most_common():

        print(
            f"  {ext:8s} : {count}"
        )

    # ========================================================
    # DATASET SOURCES
    # ========================================================

    print("\n" + "=" * 75)
    print("DATASET SOURCES")
    print("=" * 75)

    print(
        "\nDataset sources:",
        len(dataset_counts)
    )

    for name, count in dataset_counts.most_common():

        print(
            f"{name:60s} : {count}"
        )

    # ========================================================
    # EXISTING SPLITS
    # ========================================================

    print("\n" + "=" * 75)
    print("EXISTING TRAIN / VAL / TEST")
    print("=" * 75)

    if split_counts:

        for split, count in split_counts.items():

            print(
                f"{split:10s} : {count}"
            )

    else:

        print("\nNo existing split detected.")

    # ========================================================
    # ANNOTATION FILE SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("ANNOTATION FILES FOUND")
    print("=" * 75)

    total_annotation_files = sum(
        annotation_extensions.values()
    )

    print(
        "\nTotal annotation files:",
        total_annotation_files
    )

    for ext, count in annotation_extensions.most_common():

        annotation_type = ANNOTATION_EXTENSIONS[ext]

        print(
            f"  {annotation_type:20s} "
            f"{ext:8s} : {count}"
        )

    # ========================================================
    # PASS 2
    # Match images to annotation index
    # ========================================================

    print("\n" + "=" * 75)
    print("ANNOTATION MATCHING")
    print("=" * 75)

    print(
        "\nMatching annotations to images..."
    )

    inventory = []

    annotated = 0
    unannotated = 0
    ambiguous = 0

    # Progress every 10,000 images
    PROGRESS_INTERVAL = 10000

    for index, image_path in enumerate(
        image_files,
        1
    ):

        dataset = get_dataset_name(
            image_path
        )

        stem = get_stem(
            os.path.basename(image_path)
        )

        candidates = annotation_index.get(
            (dataset, stem),
            []
        )

        # ----------------------------------------------------
        # Remove duplicate annotation paths
        # ----------------------------------------------------

        candidates = list(
            dict.fromkeys(candidates)
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if len(candidates) == 0:

            has_annotation = "NO"
            unannotated += 1

        elif len(candidates) == 1:

            has_annotation = "YES"
            annotated += 1

        else:

            has_annotation = "AMBIGUOUS"
            ambiguous += 1

        # ----------------------------------------------------
        # Annotation type
        # ----------------------------------------------------

        annotation_type = get_annotation_type(
            candidates
        )

        # ----------------------------------------------------
        # Relative paths
        # ----------------------------------------------------

        project_relative_image = os.path.relpath(
            image_path,
            PROJECT_DIR
        )

        annotation_paths = []

        for annotation in candidates:

            annotation_paths.append(
                os.path.relpath(
                    annotation,
                    PROJECT_DIR
                )
            )

        # ----------------------------------------------------
        # Inventory record
        # ----------------------------------------------------

        inventory.append({

            "image_path":
                project_relative_image,

            "original_image_path":
                image_path,

            "dataset_name":
                dataset,

            "image_filename":
                os.path.basename(
                    image_path
                ),

            "image_extension":
                get_extension(
                    image_path
                ),

            "existing_split":
                get_split(
                    image_path
                ),

            "has_annotation":
                has_annotation,

            "annotation_count":
                len(candidates),

            "annotation_type":
                annotation_type,

            "annotation_files":
                ";".join(
                    annotation_paths
                )
        })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % PROGRESS_INTERVAL == 0
            or index == len(image_files)
        ):

            print(
                f"Processed "
                f"{index:,}/{len(image_files):,} images..."
            )

    # ========================================================
    # SAVE INVENTORY
    # ========================================================

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    fieldnames = [
        "image_path",
        "original_image_path",
        "dataset_name",
        "image_filename",
        "image_extension",
        "existing_split",
        "has_annotation",
        "annotation_count",
        "annotation_type",
        "annotation_files"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(inventory)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("ANNOTATION SUMMARY")
    print("=" * 75)

    print(
        "\nTotal images:",
        f"{len(image_files):,}"
    )

    print(
        "Images with annotations:",
        f"{annotated:,}"
    )

    print(
        "Images without annotations:",
        f"{unannotated:,}"
    )

    print(
        "Ambiguous annotation matches:",
        f"{ambiguous:,}"
    )

    print(
        "\nTotal annotation files:",
        f"{total_annotation_files:,}"
    )

    print("\n" + "=" * 75)
    print("OUTPUT")
    print("=" * 75)

    print("\nInventory:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 75)
    print("STEP 1 COMPLETE")
    print("=" * 75)

    print("\nOriginal dataset:")
    print(DATASET_ROOT)

    print("\nNo files were copied.")
    print("No original files were modified.")
    print("No annotations were created or deleted.")

    print(
        "\nNext step will be annotation-format analysis."
    )


if __name__ == "__main__":
    main()