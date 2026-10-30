from pathlib import Path
import csv
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_DIR = Path(
    r"D:\Blood_Smear_AI\data\rbc_candidates"
)

CSV_PATH = (
    CANDIDATE_DIR /
    "rbc_candidate_summary.csv"
)

OUTPUT_DIR = Path(
    r"D:\Blood_Smear_AI\data\rbc_selected"
)

TARGET_IMAGES = 1963


# ============================================================
# FIND CANDIDATE IMAGE
# ============================================================

def find_candidate_image(source):

    image_dir = CANDIDATE_DIR / "images"

    # First try exact original filename
    exact = image_dir / source.name

    if exact.exists():
        return exact

    # TTA may have renamed the image, for example:
    #
    # Image3542.jpg
    # Image3542_abc123.jpg
    #
    matches = list(
        image_dir.glob(
            f"{source.stem}_*"
        )
    )

    # Keep only the same image extension
    matches = [
        x for x in matches
        if x.suffix.lower() == source.suffix.lower()
    ]

    if matches:
        return matches[0]

    return None


# ============================================================
# FIND LABEL FOR THE ACTUAL CANDIDATE IMAGE
# ============================================================

def find_candidate_label(candidate_image):

    label_dir = CANDIDATE_DIR / "labels"

    # The label should normally have the same stem
    # as the actual candidate image.
    exact = (
        label_dir /
        f"{candidate_image.stem}.txt"
    )

    if exact.exists():
        return exact

    # Fallback search
    matches = list(
        label_dir.glob(
            f"{candidate_image.stem}*.txt"
        )
    )

    if matches:
        return matches[0]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SELECTING TOP RBC TTA CANDIDATES")
    print("=" * 70)

    # --------------------------------------------------------
    # Check CSV
    # --------------------------------------------------------

    if not CSV_PATH.exists():

        print(
            f"\nERROR: CSV not found:\n{CSV_PATH}"
        )

        return

    # --------------------------------------------------------
    # Read candidates
    # --------------------------------------------------------

    rows = []

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        print(
            "\nCSV columns:"
        )

        print(
            reader.fieldnames
        )

        for row in reader:

            status = (
                row.get("status", "")
                .strip()
                .upper()
            )

            if "ACCEPTED" not in status:
                continue

            try:

                confidence = float(
                    row["avg_conf"]
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            try:

                objects = int(
                    float(
                        row["RBC_objects"]
                    )
                )

            except (
                ValueError,
                TypeError
            ):

                objects = 0

            rows.append({
                "source": Path(
                    row["source_image"]
                ),
                "confidence": confidence,
                "objects": objects
            })

    print(
        f"\nAccepted TTA candidates: "
        f"{len(rows)}"
    )

    # --------------------------------------------------------
    # Sort by highest TTA confidence
    # --------------------------------------------------------

    rows.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    selected = rows[:TARGET_IMAGES]

    print(
        f"Target: "
        f"{TARGET_IMAGES}"
    )

    print(
        f"Candidates selected: "
        f"{len(selected)}"
    )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    output_image_dir = (
        OUTPUT_DIR / "images"
    )

    output_label_dir = (
        OUTPUT_DIR / "labels"
    )

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    selected_rows = []

    missing_images = 0
    missing_labels = 0

    # --------------------------------------------------------
    # Copy selected candidates
    # --------------------------------------------------------

    print("\nCopying selected candidates...")

    for rank, item in enumerate(
        selected,
        start=1
    ):

        source = item["source"]

        # ----------------------------------------------------
        # Find actual candidate image
        # ----------------------------------------------------

        candidate_image = (
            find_candidate_image(source)
        )

        if candidate_image is None:

            missing_images += 1

            continue

        # ----------------------------------------------------
        # Find label using actual candidate image name
        # ----------------------------------------------------

        candidate_label = (
            find_candidate_label(
                candidate_image
            )
        )

        if candidate_label is None:

            missing_labels += 1

            continue

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = (
            output_image_dir /
            candidate_image.name
        )

        shutil.copy2(
            candidate_image,
            destination_image
        )

        # ----------------------------------------------------
        # Copy label
        # ----------------------------------------------------

        destination_label = (
            output_label_dir /
            candidate_label.name
        )

        shutil.copy2(
            candidate_label,
            destination_label
        )

        # ----------------------------------------------------
        # Save report row
        # ----------------------------------------------------

        selected_rows.append([
            rank,
            str(source),
            candidate_image.name,
            candidate_label.name,
            f"{item['confidence']:.4f}",
            item["objects"]
        ])

        # Progress
        if rank % 100 == 0:

            print(
                f"Copied {rank}/"
                f"{len(selected)}"
            )

    # --------------------------------------------------------
    # Save selection report
    # --------------------------------------------------------

    report = (
        OUTPUT_DIR /
        "selected_rbc_summary.csv"
    )

    with open(
        report,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "rank",
            "source_image",
            "selected_image",
            "selected_label",
            "avg_conf",
            "RBC_objects"
        ])

        writer.writerows(
            selected_rows
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SELECTION COMPLETE")
    print("=" * 70)

    print(
        f"Accepted TTA candidates: "
        f"{len(rows)}"
    )

    print(
        f"Target: "
        f"{TARGET_IMAGES}"
    )

    print(
        f"Selected successfully: "
        f"{len(selected_rows)}"
    )

    print(
        f"Missing images: "
        f"{missing_images}"
    )

    print(
        f"Missing labels: "
        f"{missing_labels}"
    )

    if selected_rows:

        confidences = [
            float(row[4])
            for row in selected_rows
        ]

        print(
            f"Highest selected avg confidence: "
            f"{max(confidences):.4f}"
        )

        print(
            f"Lowest selected avg confidence: "
            f"{min(confidences):.4f}"
        )

    print(
        f"\nOutput:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        f"\nReport:"
    )

    print(
        report
    )

    print("\nDONE.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()