from pathlib import Path
from collections import Counter

# ============================================================
# FINAL DATASET BALANCE ANALYSIS
# ============================================================

DATASET = Path(r"D:\Blood_Smear_AI\data\final_dataset")

TRAIN_LABELS = DATASET / "labels" / "train"

CLASSES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet",
}


def main():

    print("=" * 75)
    print("FINAL TRAINING DATASET BALANCE ANALYSIS")
    print("=" * 75)

    label_files = list(
        TRAIN_LABELS.glob("*.txt")
    )

    print(f"\nTraining images/labels : {len(label_files):,}")

    object_counts = Counter()

    image_counts = Counter()

    # Number of objects per image
    objects_per_image = Counter()

    for label_file in label_files:

        classes_in_image = set()
        total_objects = 0

        try:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) < 5:
                        continue

                    try:
                        cid = int(parts[0])
                    except:
                        continue

                    if cid not in CLASSES:
                        continue

                    object_counts[cid] += 1
                    classes_in_image.add(cid)
                    total_objects += 1

        except Exception:
            continue

        for cid in classes_in_image:
            image_counts[cid] += 1

        objects_per_image[total_objects] += 1

    # --------------------------------------------------------
    # Object counts
    # --------------------------------------------------------

    total_objects = sum(
        object_counts.values()
    )

    print("\n")
    print("OBJECT COUNTS")
    print("-" * 75)

    print(
        f"{'CLASS':<15}"
        f"{'OBJECTS':>15}"
        f"{'% OF OBJECTS':>18}"
        f"{'IMAGES':>15}"
    )

    print("-" * 75)

    for cid, name in CLASSES.items():

        count = object_counts[cid]

        percentage = (
            count / total_objects * 100
            if total_objects
            else 0
        )

        print(
            f"{name:<15}"
            f"{count:>15,}"
            f"{percentage:>17.2f}%"
            f"{image_counts[cid]:>15,}"
        )

    print("-" * 75)

    print(
        f"{'TOTAL':<15}"
        f"{total_objects:>15,}"
    )

    # --------------------------------------------------------
    # Image-level distribution
    # --------------------------------------------------------

    print("\n")
    print("IMAGE-LEVEL CLASS PRESENCE")
    print("-" * 75)

    for cid, name in CLASSES.items():

        percentage = (
            image_counts[cid]
            / len(label_files)
            * 100
            if label_files
            else 0
        )

        print(
            f"{name:<15}"
            f"{image_counts[cid]:>10,} images"
            f" ({percentage:.2f}% of training images)"
        )

    # --------------------------------------------------------
    # Imbalance ratios
    # --------------------------------------------------------

    print("\n")
    print("CLASS IMBALANCE")
    print("-" * 75)

    rbc = object_counts[0]
    wbc = object_counts[1]
    platelet = object_counts[2]

    print(
        f"WBC / RBC       : {wbc / rbc:.2f}x"
        if rbc else
        "WBC / RBC       : N/A"
    )

    print(
        f"RBC / Platelet   : {rbc / platelet:.2f}x"
        if platelet else
        "RBC / Platelet   : N/A"
    )

    print(
        f"WBC / Platelet   : {wbc / platelet:.2f}x"
        if platelet else
        "WBC / Platelet   : N/A"
    )

    # --------------------------------------------------------
    # Recommended augmentation
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("RECOMMENDATION")
    print("=" * 75)

    # WBC is already dominant.
    if wbc > rbc * 3:

        print(
            "\nWBC is strongly dominant."
        )

        print(
            "DO NOT augment WBC."
        )

    # RBC
    if rbc < wbc * 0.30:

        print(
            "RBC needs additional targeted augmentation."
        )

    else:

        print(
            "RBC augmentation level is acceptable."
        )

    # Platelet
    if platelet < rbc * 0.70:

        print(
            "Platelet may benefit from targeted augmentation."
        )

    else:

        print(
            "Platelet augmentation level is acceptable."
        )

    print("\n")
    print("IMPORTANT:")
    print(
        "Only TRAIN images should receive additional augmentation."
    )
    print(
        "Validation and TEST images will NOT be modified."
    )

    print("\n")
    print("=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()