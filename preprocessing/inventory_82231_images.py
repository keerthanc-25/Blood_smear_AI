
from pathlib import Path
import hashlib
import csv
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

SOURCE = Path(r"D:\blood smear images")

ORIGINAL_DATASET = Path(
    r"D:\Blood_Smear_AI\data\yolo_dataset"
)

PSEUDO_ROOT = Path(
    r"D:\Blood_Smear_AI\data\pseudo_labeled_new"
)

RBC_SECOND = Path(
    r"D:\Blood_Smear_AI\data\rbc_second_pass"
)

RBC_SELECTED = Path(
    r"D:\Blood_Smear_AI\data\rbc_selected"
)

OUTPUT = Path(
    r"D:\Blood_Smear_AI\data\dataset_inventory_82231"
)

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

# Original class mapping
# 0 = RBC
# 1 = WBC
# 2 = Platelet
# 3 = Malaria_Parasite


# ============================================================
# HASH
# ============================================================

def image_hash(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# GET ALL SOURCE IMAGES
# ============================================================

def get_source_images():

    print("=" * 70)
    print("SCANNING SOURCE DATASET")
    print("=" * 70)

    images = []

    for path in SOURCE.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in EXTENSIONS
            and "_MACOSX" not in path.parts
        ):
            images.append(path)

    print(
        f"Source images found: {len(images)}"
    )

    return images


# ============================================================
# BUILD SOURCE HASH MAP
# ============================================================

def build_source_hashes(images):

    print("\n" + "=" * 70)
    print("HASHING SOURCE IMAGES")
    print("=" * 70)

    hashes = {}
    errors = 0

    for i, image in enumerate(images, 1):

        try:
            h = image_hash(image)
            hashes[h] = image

        except Exception as e:

            errors += 1
            print(
                f"[ERROR] {image}: {e}"
            )

        if i % 10000 == 0:

            print(
                f"Hashed {i}/{len(images)}"
            )

    print(
        f"Hashing complete. Errors: {errors}"
    )

    return hashes


# ============================================================
# FIND LABELS ASSOCIATED WITH AN IMAGE
# ============================================================

def possible_labels(image, root):

    candidates = []

    # Same folder, same stem
    candidates.append(
        image.with_suffix(".txt")
    )

    # labels folder parallel to images
    parts = list(image.parts)

    for i, part in enumerate(parts):

        if part.lower() == "images":

            label_parts = parts[:]

            label_parts[i] = "labels"

            label_path = Path(
                *label_parts
            ).with_suffix(".txt")

            candidates.append(label_path)

    # Dataset-wide search by filename
    candidates.append(
        root / "labels" / f"{image.stem}.txt"
    )

    return candidates


# ============================================================
# READ LABEL CLASSES
# ============================================================

def read_classes(label):

    classes = set()

    try:

        with open(
            label,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if not parts:
                    continue

                try:

                    cls = int(parts[0])
                    classes.add(cls)

                except ValueError:
                    pass

    except Exception:
        pass

    return classes


# ============================================================
# COLLECT ORIGINAL ANNOTATED HASHES
# ============================================================

def collect_original_annotated():

    print("\n" + "=" * 70)
    print("CHECKING ORIGINAL ANNOTATED DATA")
    print("=" * 70)

    annotated = {}
    malaria = {}
    other = {}

    label_files = list(
        ORIGINAL_DATASET.rglob("*.txt")
    )

    # Build image lookup by stem/name
    image_files = [
        p for p in ORIGINAL_DATASET.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in EXTENSIONS
        )
    ]

    by_name = defaultdict(list)

    for image in image_files:
        by_name[image.name].append(image)

    for label in label_files:

        if label.name == "classes.txt":
            continue

        image_candidates = by_name.get(
            f"{label.stem}.jpg",
            []
        )

        for ext in EXTENSIONS:

            image_candidates.extend(
                by_name.get(
                    f"{label.stem}{ext}",
                    []
                )
            )

        # Remove duplicates
        image_candidates = list(
            dict.fromkeys(image_candidates)
        )

        if not image_candidates:
            continue

        classes = read_classes(label)

        for image in image_candidates:

            try:
                h = image_hash(image)
            except Exception:
                continue

            annotated[h] = image

            if 3 in classes:
                malaria[h] = image

            other_classes = classes - {
                0, 1, 2, 3
            }

            if other_classes:
                other[h] = image

    print(
        f"Original annotated images: "
        f"{len(annotated)}"
    )

    print(
        f"Original images containing malaria: "
        f"{len(malaria)}"
    )

    print(
        f"Original images containing other classes: "
        f"{len(other)}"
    )

    return annotated, malaria, other


# ============================================================
# COLLECT PSEUDO-LABEL HASHES
# ============================================================

def collect_pseudo_folder(root):

    hashes = {}
    errors = 0

    image_dir = root / "images"
    label_dir = root / "labels"

    if not image_dir.exists():

        return hashes, errors

    for image in image_dir.iterdir():

        if (
            not image.is_file()
            or image.suffix.lower()
            not in EXTENSIONS
        ):
            continue

        label = (
            label_dir /
            f"{image.stem}.txt"
        )

        if not label.exists():
            continue

        try:

            h = image_hash(image)
            hashes[h] = image

        except Exception:

            errors += 1

    return hashes, errors


# ============================================================
# COLLECT ALL PSEUDO ANNOTATIONS
# ============================================================

def collect_pseudo():

    print("\n" + "=" * 70)
    print("CHECKING PSEUDO-ANNOTATED DATA")
    print("=" * 70)

    folders = {
        "pseudo_rbc":
            PSEUDO_ROOT / "rbc",

        "pseudo_wbc_platelet":
            PSEUDO_ROOT / "wbc_platelet",

        "rbc_second_pass":
            RBC_SECOND,

        "rbc_selected":
            RBC_SELECTED,
    }

    all_pseudo = {}
    by_group = {}

    for name, folder in folders.items():

        hashes, errors = collect_pseudo_folder(
            folder
        )

        by_group[name] = hashes

        print(
            f"{name}: {len(hashes)} images"
        )

        if errors:
            print(
                f"  hash errors: {errors}"
            )

        for h, path in hashes.items():

            all_pseudo[h] = path

    print(
        f"\nUnique pseudo-annotated images: "
        f"{len(all_pseudo)}"
    )

    return all_pseudo, by_group


# ============================================================
# CLASSIFY SOURCE IMAGES
# ============================================================

def classify_source(
    source_hashes,
    original,
    malaria,
    other,
    pseudo
):

    print("\n" + "=" * 70)
    print("CLASSIFYING SOURCE IMAGES")
    print("=" * 70)

    counts = {
        "original_annotated": 0,
        "pseudo_annotated": 0,
        "malaria": 0,
        "other_cell_types": 0,
        "remaining_unclassified": 0
    }

    rows = []

    for h, image in source_hashes.items():

        # Original annotated has highest priority
        if h in original:

            if h in malaria:
                category = "original_annotated_malaria"
            elif h in other:
                category = "original_annotated_other_class"
            else:
                category = "original_annotated"

        elif h in pseudo:

            category = "pseudo_annotated"

        else:

            # These are not necessarily malaria.
            # They are simply not identified by our
            # current annotation/pseudo-label inventories.
            category = "remaining_unclassified"

        if category == "original_annotated":
            counts["original_annotated"] += 1

        elif category == "pseudo_annotated":
            counts["pseudo_annotated"] += 1

        elif category == "remaining_unclassified":
            counts["remaining_unclassified"] += 1

        # Malaria and other classes are tracked
        # separately below.

        rows.append([
            str(image),
            h,
            category
        ])

    # Count malaria/other as independent properties
    # of original annotated images.
    counts["malaria"] = len(malaria)

    counts["other_cell_types"] = len(other)

    return counts, rows


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    source_images = get_source_images()

    source_hashes = build_source_hashes(
        source_images
    )

    original, malaria, other = (
        collect_original_annotated()
    )

    pseudo, pseudo_groups = (
        collect_pseudo()
    )

    counts, rows = classify_source(
        source_hashes,
        original,
        malaria,
        other,
        pseudo
    )

    # --------------------------------------------------------
    # IMPORTANT: create mutually exclusive categories
    # --------------------------------------------------------

    # Original annotated images that contain malaria
    original_malaria = (
        set(original.keys())
        & set(malaria.keys())
    )

    # Original annotated images with non-3-class labels
    original_other = (
        set(original.keys())
        & set(other.keys())
    )

    # Original annotated images that are clean 3-class
    original_clean = (
        set(original.keys())
        - original_malaria
        - original_other
    )

    # Pseudo images that are not already original
    pseudo_unique = (
        set(pseudo.keys())
        - set(original.keys())
    )

    # Everything else
    remaining = (
        set(source_hashes.keys())
        - set(original.keys())
        - pseudo_unique
    )

    exclusive = {
        "original_annotated_3class":
            len(original_clean),

        "original_annotated_malaria":
            len(original_malaria),

        "original_annotated_other":
            len(original_other),

        "pseudo_annotated":
            len(pseudo_unique),

        "remaining_unclassified":
            len(remaining)
    }

    total = sum(
        exclusive.values()
    )

    print("\n" + "=" * 70)
    print("MASTER DATASET INVENTORY")
    print("=" * 70)

    print(
        f"TOTAL SOURCE IMAGES: "
        f"{len(source_hashes)}"
    )

    print("\nMUTUALLY EXCLUSIVE CATEGORIES")

    for name, count in exclusive.items():

        print(
            f"{name:35s}: {count}"
        )

    print(
        f"{'CHECK TOTAL':35s}: {total}"
    )

    print(
        f"{'DIFFERENCE':35s}: "
        f"{len(source_hashes) - total}"
    )

    print("\nAdditional information:")

    print(
        f"Original annotated total: "
        f"{len(original)}"
    )

    print(
        f"Original malaria-containing: "
        f"{len(original_malaria)}"
    )

    print(
        f"Original other-class-containing: "
        f"{len(original_other)}"
    )

    print(
        f"Unique pseudo-annotated: "
        f"{len(pseudo_unique)}"
    )

    print("\nPseudo-label groups:")

    for name, group in pseudo_groups.items():

        print(
            f"{name:25s}: {len(group)}"
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT /
        "source_image_inventory.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "source_image",
            "sha256",
            "classification"
        ])

        for row in rows:
            writer.writerow(row)

    summary_path = (
        OUTPUT /
        "inventory_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "category",
            "image_count"
        ])

        for name, count in exclusive.items():

            writer.writerow([
                name,
                count
            ])

        writer.writerow([
            "TOTAL",
            total
        ])

    print("\nReports saved to:")
    print(OUTPUT)

    print("\nDONE.")


if __name__ == "__main__":
    main()
