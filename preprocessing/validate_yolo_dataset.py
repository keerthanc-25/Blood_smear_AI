import os
import csv
from collections import Counter

PROJECT_DIR = r"D:\Blood_Smear_AI"

MANIFEST = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "annotation_manifest.csv"
)

REPORT = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "validation_report.csv"
)

VALID_CLASSES = {0, 1, 2, 3}

CLASS_NAMES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet",
    3: "Malaria_Parasite"
}


def validate_label_file(path):

    valid = True
    objects = 0
    class_counts = Counter()
    errors = []

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()

        if not lines:

            return False, 0, class_counts, [
                "EMPTY_LABEL"
            ]

        for line_number, line in enumerate(
            lines,
            1
        ):

            parts = line.strip().split()

            if len(parts) != 5:

                valid = False

                errors.append(
                    f"LINE_{line_number}_WRONG_FIELDS"
                )

                continue

            try:

                class_id = int(parts[0])

                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:

                valid = False

                errors.append(
                    f"LINE_{line_number}_NON_NUMERIC"
                )

                continue

            if class_id not in VALID_CLASSES:

                valid = False

                errors.append(
                    f"LINE_{line_number}_INVALID_CLASS"
                )

                continue

            if not (
                0 <= xc <= 1
                and 0 <= yc <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):

                valid = False

                errors.append(
                    f"LINE_{line_number}_INVALID_BOX"
                )

                continue

            objects += 1
            class_counts[class_id] += 1

    except Exception as e:

        return False, 0, class_counts, [
            f"READ_ERROR:{e}"
        ]

    return valid, objects, class_counts, errors


def main():

    print("=" * 70)
    print("STEP 4 - VALIDATE YOLO DATASET")
    print("=" * 70)

    if not os.path.exists(MANIFEST):

        print("\nERROR:")
        print("Manifest not found:")
        print(MANIFEST)

        return

    with open(
        MANIFEST,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    print(
        "\nManifest records:",
        f"{len(rows):,}"
    )

    valid_images = 0
    invalid_images = 0
    missing_images = 0
    missing_labels = 0

    total_objects = 0

    class_counts = Counter()

    report = []

    # Prevent duplicate image counting
    unique_images = set()

    for index, row in enumerate(
        rows,
        1
    ):

        image_file = row["image_file"]
        label_file = row["label_file"]

        unique_images.add(
            os.path.normcase(
                os.path.normpath(
                    image_file
                )
            )
        )

        image_exists = os.path.isfile(
            image_file
        )

        label_exists = os.path.isfile(
            label_file
        )

        if not image_exists:

            missing_images += 1

            report.append({
                "image": image_file,
                "label": label_file,
                "status": "MISSING_IMAGE",
                "objects": 0,
                "errors": "MISSING_IMAGE"
            })

            continue

        if not label_exists:

            missing_labels += 1

            report.append({
                "image": image_file,
                "label": label_file,
                "status": "MISSING_LABEL",
                "objects": 0,
                "errors": "MISSING_LABEL"
            })

            continue

        valid, objects, counts, errors = (
            validate_label_file(
                label_file
            )
        )

        total_objects += objects

        for class_id, count in counts.items():

            class_counts[class_id] += count

        if valid:

            valid_images += 1
            status = "VALID"

        else:

            invalid_images += 1
            status = "INVALID"

        report.append({
            "image": image_file,
            "label": label_file,
            "status": status,
            "objects": objects,
            "errors": ";".join(errors)
        })

        if (
            index % 1000 == 0
            or index == len(rows)
        ):

            print(
                f"Validated "
                f"{index:,}/{len(rows):,}"
            )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT),
        exist_ok=True
    )

    with open(
        REPORT,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "label",
                "status",
                "objects",
                "errors"
            ]
        )

        writer.writeheader()
        writer.writerows(report)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        "\nManifest records       :",
        f"{len(rows):,}"
    )

    print(
        "Unique images          :",
        f"{len(unique_images):,}"
    )

    print(
        "Valid image/labels     :",
        f"{valid_images:,}"
    )

    print(
        "Invalid labels         :",
        f"{invalid_images:,}"
    )

    print(
        "Missing images         :",
        f"{missing_images:,}"
    )

    print(
        "Missing labels         :",
        f"{missing_labels:,}"
    )

    print(
        "Total valid objects    :",
        f"{total_objects:,}"
    )

    print("\nCLASS DISTRIBUTION")

    for class_id in range(4):

        print(
            f"{class_id} = "
            f"{CLASS_NAMES[class_id]:18s} : "
            f"{class_counts[class_id]:,}"
        )

    print("\nREPORT:")
    print(REPORT)

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()