from pathlib import Path
import hashlib
import shutil


# ============================================================
# PATHS
# ============================================================

VERIFIED_DATASET = Path(
    r"D:\Blood_Smear_AI\data\yolo_dataset_3class"
)

PSEUDO_RBC = Path(
    r"D:\Blood_Smear_AI\data\pseudo_labeled_new\rbc"
)

PSEUDO_WBC_PLATELET = Path(
    r"D:\Blood_Smear_AI\data\pseudo_labeled_new\wbc_platelet"
)

RBC_SECOND_PASS = Path(
    r"D:\Blood_Smear_AI\data\rbc_second_pass"
)

RBC_SELECTED = Path(
    r"D:\Blood_Smear_AI\data\rbc_selected"
)

FINAL_DATASET = Path(
    r"D:\Blood_Smear_AI\data\yolo_dataset_final_3class"
)

EXTENSIONS = {
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
# LABEL COUNT
# ============================================================

def count_objects(label_path):

    counts = {
        0: 0,
        1: 0,
        2: 0
    }

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if not parts:
                continue

            try:

                cls = int(parts[0])

                if cls in counts:
                    counts[cls] += 1

            except ValueError:
                pass

    return counts


# ============================================================
# CREATE FINAL STRUCTURE
# ============================================================

def create_structure():

    for split in [
        "train",
        "val",
        "test"
    ]:

        (
            FINAL_DATASET /
            "images" /
            split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            FINAL_DATASET /
            "labels" /
            split
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY ORIGINAL VERIFIED DATASET
# ============================================================

def copy_verified(existing_hashes):

    print("\n" + "=" * 70)
    print("COPYING VERIFIED DATASET")
    print("=" * 70)

    for split in [
        "train",
        "val",
        "test"
    ]:

        source_images = (
            VERIFIED_DATASET /
            "images" /
            split
        )

        source_labels = (
            VERIFIED_DATASET /
            "labels" /
            split
        )

        destination_images = (
            FINAL_DATASET /
            "images" /
            split
        )

        destination_labels = (
            FINAL_DATASET /
            "labels" /
            split
        )

        added = 0

        for image in source_images.iterdir():

            if (
                not image.is_file()
                or image.suffix.lower()
                not in EXTENSIONS
            ):
                continue

            label = (
                source_labels /
                f"{image.stem}.txt"
            )

            if not label.exists():

                print(
                    f"[WARNING] Missing label: {image}"
                )

                continue

            h = file_hash(image)

            if h in existing_hashes:
                continue

            shutil.copy2(
                image,
                destination_images /
                image.name
            )

            shutil.copy2(
                label,
                destination_labels /
                label.name
            )

            existing_hashes.add(h)

            added += 1

        print(
            f"{split}: {added} verified images"
        )


# ============================================================
# ADD PSEUDO DATASET TO TRAIN
# ============================================================

def add_pseudo_dataset(
    source_root,
    prefix,
    existing_hashes
):

    print("\n" + "=" * 70)
    print(
        f"ADDING PSEUDO DATASET: {prefix}"
    )
    print("=" * 70)

    source_images = (
        source_root /
        "images"
    )

    source_labels = (
        source_root /
        "labels"
    )

    destination_images = (
        FINAL_DATASET /
        "images" /
        "train"
    )

    destination_labels = (
        FINAL_DATASET /
        "labels" /
        "train"
    )

    if not source_images.exists():

        print(
            f"[ERROR] Images folder not found:"
        )

        print(source_images)

        return

    added = 0
    duplicates = 0
    missing_labels = 0

    objects = {
        0: 0,
        1: 0,
        2: 0
    }

    for image in source_images.iterdir():

        if (
            not image.is_file()
            or image.suffix.lower()
            not in EXTENSIONS
        ):
            continue

        label = (
            source_labels /
            f"{image.stem}.txt"
        )

        if not label.exists():

            missing_labels += 1

            continue

        h = file_hash(image)

        if h in existing_hashes:

            duplicates += 1

            continue

        # Unique name prevents filename collisions
        new_name = (
            f"{prefix}_{image.name}"
        )

        destination_image = (
            destination_images /
            new_name
        )

        destination_label = (
            destination_labels /
            f"{Path(new_name).stem}.txt"
        )

        shutil.copy2(
            image,
            destination_image
        )

        shutil.copy2(
            label,
            destination_label
        )

        existing_hashes.add(h)

        counts = count_objects(label)

        objects[0] += counts[0]
        objects[1] += counts[1]
        objects[2] += counts[2]

        added += 1

    print(
        f"Images added:       {added}"
    )

    print(
        f"RBC objects:        {objects[0]}"
    )

    print(
        f"WBC objects:        {objects[1]}"
    )

    print(
        f"Platelet objects:   {objects[2]}"
    )

    print(
        f"Duplicates skipped: {duplicates}"
    )

    print(
        f"Missing labels:     {missing_labels}"
    )


# ============================================================
# DATA.YAML
# ============================================================

def create_yaml():

    yaml_path = (
        FINAL_DATASET /
        "data.yaml"
    )

    content = f"""path: {FINAL_DATASET.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 3

names:
  0: RBC
  1: WBC
  2: Platelet
"""

    yaml_path.write_text(
        content,
        encoding="utf-8"
    )

    print(
        "\ndata.yaml created."
    )


# ============================================================
# FINAL COUNT
# ============================================================

def final_report():

    print("\n" + "=" * 70)
    print("FINAL DATASET COUNT")
    print("=" * 70)

    total_images = 0

    total_objects = {
        0: 0,
        1: 0,
        2: 0
    }

    for split in [
        "train",
        "val",
        "test"
    ]:

        image_dir = (
            FINAL_DATASET /
            "images" /
            split
        )

        label_dir = (
            FINAL_DATASET /
            "labels" /
            split
        )

        image_count = 0

        objects = {
            0: 0,
            1: 0,
            2: 0
        }

        for image in image_dir.iterdir():

            if (
                not image.is_file()
                or image.suffix.lower()
                not in EXTENSIONS
            ):
                continue

            image_count += 1

            label = (
                label_dir /
                f"{image.stem}.txt"
            )

            if not label.exists():
                continue

            counts = count_objects(label)

            objects[0] += counts[0]
            objects[1] += counts[1]
            objects[2] += counts[2]

        total_images += image_count

        total_objects[0] += objects[0]
        total_objects[1] += objects[1]
        total_objects[2] += objects[2]

        print(f"\n{split.upper()}")

        print(
            f"Images:   {image_count}"
        )

        print(
            f"RBC:      {objects[0]}"
        )

        print(
            f"WBC:      {objects[1]}"
        )

        print(
            f"Platelet: {objects[2]}"
        )

    print("\n" + "-" * 70)

    print(
        f"TOTAL IMAGES:          {total_images}"
    )

    print(
        f"TOTAL RBC OBJECTS:     {total_objects[0]}"
    )

    print(
        f"TOTAL WBC OBJECTS:     {total_objects[1]}"
    )

    print(
        f"TOTAL PLATELET OBJECTS: {total_objects[2]}"
    )

    print(
        "\nFINAL DATASET:"
    )

    print(
        FINAL_DATASET
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL 3-CLASS DATASET CREATION")
    print("=" * 70)

    print(
        "\nOriginal verified train/val/test will be preserved."
    )

    print(
        "All pseudo-labelled data will be added to TRAIN."
    )

    # --------------------------------------------------------
    # Delete previous final dataset ONLY
    # --------------------------------------------------------

    if FINAL_DATASET.exists():

        print(
            f"\nExisting final dataset found:"
        )

        print(
            FINAL_DATASET
        )

        answer = input(
            "\nDelete and rebuild it? [y/N]: "
        ).strip().lower()

        if answer != "y":

            print(
                "Stopped."
            )

            return

        shutil.rmtree(
            FINAL_DATASET
        )

    create_structure()

    existing_hashes = set()

    # --------------------------------------------------------
    # 1. VERIFIED DATASET
    # --------------------------------------------------------

    copy_verified(
        existing_hashes
    )

    # --------------------------------------------------------
    # 2. ORIGINAL RBC PSEUDO LABELS
    # --------------------------------------------------------

    add_pseudo_dataset(
        PSEUDO_RBC,
        "pseudo_rbc",
        existing_hashes
    )

    # --------------------------------------------------------
    # 3. WBC + PLATELET PSEUDO LABELS
    # --------------------------------------------------------

    add_pseudo_dataset(
        PSEUDO_WBC_PLATELET,
        "pseudo_wbc_platelet",
        existing_hashes
    )

    # --------------------------------------------------------
    # 4. RBC SECOND PASS
    # --------------------------------------------------------

    add_pseudo_dataset(
        RBC_SECOND_PASS,
        "rbc_second",
        existing_hashes
    )

    # --------------------------------------------------------
    # 5. TOP TTA RBC CANDIDATES
    # --------------------------------------------------------

    add_pseudo_dataset(
        RBC_SELECTED,
        "rbc_tta",
        existing_hashes
    )

    # --------------------------------------------------------
    # 6. YAML
    # --------------------------------------------------------

    create_yaml()

    # --------------------------------------------------------
    # 7. FINAL VERIFICATION
    # --------------------------------------------------------

    final_report()

    print("\n" + "=" * 70)
    print("COMBINATION COMPLETE")
    print("=" * 70)

    print(
        "\nNO SOURCE DATASETS WERE DELETED."
    )

    print(
        "Verify the final counts before cleanup."
    )


if __name__ == "__main__":
    main()