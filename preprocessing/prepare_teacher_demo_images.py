from pathlib import Path
import random
import shutil

PROJECT = Path(r"D:\Blood_Smear_AI")

# Main dataset used to train the three-class YOLO model
DATASET = PROJECT / "data" / "final_dataset"

# Existing demo collection
OUTPUT = PROJECT / "teacher_demo_images"

TARGET_PER_CATEGORY = 100
SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

# YOLO class mapping
CLASSES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet",
}

random.seed(SEED)


def read_image_labels(split):
    """Read images and their corresponding YOLO annotations."""
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        print(f"Skipping missing split: {split}")
        return []

    records = []

    for image in image_dir.iterdir():
        if not image.is_file():
            continue

        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            continue

        try:
            lines = label.read_text(encoding="utf-8").splitlines()
            class_ids = set()

            for line in lines:
                parts = line.strip().split()

                if parts:
                    try:
                        class_ids.add(int(float(parts[0])))
                    except ValueError:
                        continue

            if class_ids:
                records.append((image, class_ids))

        except OSError as exc:
            print(f"Could not read {label}: {exc}")

    return records


def copy_category(category_name, records, target_count):
    destination = OUTPUT / category_name
    destination.mkdir(parents=True, exist_ok=True)

    existing = [
        p for p in destination.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    existing_count = len(existing)

    if existing_count >= target_count:
        print(f"{category_name}: already has {existing_count} images")
        return existing_count

    needed = target_count - existing_count
    random.shuffle(records)

    copied = 0

    for image, _ in records:
        if copied >= needed:
            break

        # Use unique names so existing demo images are not overwritten.
        target = destination / f"YOLO_{image.name}"

        if target.exists():
            continue

        try:
            shutil.copy2(image, target)
            copied += 1
        except OSError as exc:
            print(f"Copy failed for {image}: {exc}")

    total = existing_count + copied

    print(
        f"{category_name}: existing={existing_count}, "
        f"added={copied}, total={total}"
    )

    if total < target_count:
        print(
            f"WARNING: Only {total} images available for {category_name}"
        )

    return total


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Scanning the main YOLO dataset...")

    # Use the test split first for demonstration.
    # Validation and training splits are fallback sources.
    records_by_split = {
        split: read_image_labels(split)
        for split in ("test", "val", "train")
    }

    for class_id, class_name in CLASSES.items():
        print(f"\nPreparing {class_name} examples...")

        # Prefer test images, then validation, then training.
        selected = []
        seen = set()

        for split in ("test", "val", "train"):
            for image, class_ids in records_by_split[split]:
                if class_id in class_ids:
                    resolved = str(image.resolve()).lower()

                    if resolved not in seen:
                        seen.add(resolved)
                        selected.append((image, class_ids))

        category_folder = {
            "RBC": "01_RBC_Morphology",
            "WBC": "02_WBC_Examples",
            "Platelet": "03_Platelet_Examples",
        }[class_name]

        copy_category(
            category_folder,
            selected,
            TARGET_PER_CATEGORY,
        )

    print("\nDemo image collection finished.")
    print(f"Output: {OUTPUT}")

    print("\nFinal image counts:")
    for folder in (
        "01_RBC_Morphology",
        "02_WBC_Examples",
        "03_Platelet_Examples",
    ):
        path = OUTPUT / folder
        count = sum(
            1 for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ) if path.exists() else 0

        print(f"{folder}: {count}")


if __name__ == "__main__":
    main()