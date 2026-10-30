import os
import shutil
from collections import Counter

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"D:\Blood_Smear_AI"

SOURCE_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "yolo_dataset"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "yolo_dataset_3class"
)

# ============================================================
# CLASSES TO KEEP
# ============================================================

# Original:
# 0 = RBC
# 1 = WBC
# 2 = Platelet
# 3 = Malaria_Parasite

VALID_CLASSES = {0, 1, 2}

# ============================================================
# PROCESS DATASET
# ============================================================

def process_split(split):

    source_images = os.path.join(
        SOURCE_DIR, "images", split
    )

    source_labels = os.path.join(
        SOURCE_DIR, "labels", split
    )

    output_images = os.path.join(
        OUTPUT_DIR, "images", split
    )

    output_labels = os.path.join(
        OUTPUT_DIR, "labels", split
    )

    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    total_images = 0
    kept_images = 0
    skipped_images = 0

    object_counts = Counter()

    # --------------------------------------------------------
    # Check source folders
    # --------------------------------------------------------

    if not os.path.exists(source_images):
        print(f"ERROR: Image folder not found:")
        print(source_images)
        return 0, 0, 0, object_counts

    if not os.path.exists(source_labels):
        print(f"ERROR: Label folder not found:")
        print(source_labels)
        return 0, 0, 0, object_counts

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for filename in os.listdir(source_images):

        image_path = os.path.join(
            source_images,
            filename
        )

        if not os.path.isfile(image_path):
            continue

        total_images += 1

        base_name = os.path.splitext(filename)[0]

        label_filename = base_name + ".txt"

        label_path = os.path.join(
            source_labels,
            label_filename
        )

        # No label
        if not os.path.exists(label_path):
            skipped_images += 1
            continue

        new_labels = []

        # ----------------------------------------------------
        # Read annotations
        # ----------------------------------------------------

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            # Ignore malaria
            if class_id not in VALID_CLASSES:
                continue

            # Keep RBC/WBC/Platelet annotation
            new_labels.append(line.strip())

            object_counts[class_id] += 1

        # ----------------------------------------------------
        # Skip images containing ONLY malaria
        # ----------------------------------------------------

        if len(new_labels) == 0:
            skipped_images += 1
            continue

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = os.path.join(
            output_images,
            filename
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        # ----------------------------------------------------
        # Save filtered labels
        # ----------------------------------------------------

        destination_label = os.path.join(
            output_labels,
            label_filename
        )

        with open(destination_label, "w") as f:
            f.write("\n".join(new_labels))

        kept_images += 1

    return (
        total_images,
        kept_images,
        skipped_images,
        object_counts
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("CREATING 3-CLASS RBC/WBC/PLATELET DATASET")
    print("=" * 65)

    print("\nSOURCE:")
    print(SOURCE_DIR)

    print("\nOUTPUT:")
    print(OUTPUT_DIR)

    print("\nClasses:")
    print("0 = RBC")
    print("1 = WBC")
    print("2 = Platelet")
    print("3 = Malaria_Parasite -> EXCLUDED")

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_DIR):

        print("\nWARNING:")
        print("Output folder already exists:")
        print(OUTPUT_DIR)

        print("\nDelete the incomplete output folder before")
        print("running this script again.")

        return

    os.makedirs(OUTPUT_DIR)

    # --------------------------------------------------------
    # Overall counters
    # --------------------------------------------------------

    total_images_all = 0
    kept_images_all = 0
    skipped_images_all = 0

    total_objects = Counter()

    # --------------------------------------------------------
    # Process train / val / test
    # --------------------------------------------------------

    for split in ["train", "val", "test"]:

        print("\n" + "-" * 65)
        print(f"PROCESSING: {split.upper()}")
        print("-" * 65)

        total, kept, skipped, counts = process_split(split)

        total_images_all += total
        kept_images_all += kept
        skipped_images_all += skipped

        total_objects.update(counts)

        print(f"Images found   : {total}")
        print(f"Images kept    : {kept}")
        print(f"Images skipped : {skipped}")

        print("\nObjects kept:")
        print(f"RBC      : {counts[0]}")
        print(f"WBC      : {counts[1]}")
        print(f"Platelet : {counts[2]}")

    # --------------------------------------------------------
    # Create data.yaml
    # --------------------------------------------------------

    yaml_path = os.path.join(
        OUTPUT_DIR,
        "data.yaml"
    )

    yaml_content = f"""path: {OUTPUT_DIR.replace(chr(92), '/')}
train: images/train
val: images/val
test: images/test

nc: 3

names:
  0: RBC
  1: WBC
  2: Platelet
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("3-CLASS DATASET CREATED SUCCESSFULLY")
    print("=" * 65)

    print(f"\nTotal images found   : {total_images_all}")
    print(f"Total images kept    : {kept_images_all}")
    print(f"Total images skipped : {skipped_images_all}")

    print("\nTOTAL OBJECTS:")
    print(f"RBC      : {total_objects[0]}")
    print(f"WBC      : {total_objects[1]}")
    print(f"Platelet : {total_objects[2]}")

    print("\nMalaria parasite annotations were excluded.")

    print("\nDATA YAML:")
    print(yaml_path)

    print("\nDataset:")
    print(OUTPUT_DIR)

    print("\nNext step:")
    print("Verify the dataset before training.")


if __name__ == "__main__":
    main()