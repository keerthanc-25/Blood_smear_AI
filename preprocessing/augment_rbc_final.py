from pathlib import Path
import cv2
import random
import numpy as np

# ============================================================
# FINAL RBC TARGETED AUGMENTATION
# Adds exactly 3,000 RBC-focused images to TRAIN
# Validation and TEST are NOT touched.
# ============================================================

DATASET = Path(r"D:\Blood_Smear_AI\data\final_dataset")

TRAIN_IMAGES = DATASET / "images" / "train"
TRAIN_LABELS = DATASET / "labels" / "train"

TARGET_AUGMENTATIONS = 3000

RBC_CLASS = 0

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# READ LABEL
# ============================================================

def read_label(path):

    objects = []

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:

                    cid = int(parts[0])

                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except:
                    continue

                if cid not in [0, 1, 2]:
                    continue

                objects.append(
                    [cid, xc, yc, w, h]
                )

    except:
        return []

    return objects


# ============================================================
# WRITE LABEL
# ============================================================

def write_label(path, objects):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for obj in objects:

            cid, xc, yc, w, h = obj

            f.write(
                f"{cid} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


# ============================================================
# HORIZONTAL FLIP
# ============================================================

def horizontal_flip(image, objects):

    image = cv2.flip(
        image,
        1
    )

    new_objects = []

    for obj in objects:

        cid, xc, yc, w, h = obj

        xc = 1.0 - xc

        new_objects.append(
            [cid, xc, yc, w, h]
        )

    return image, new_objects


# ============================================================
# VERTICAL FLIP
# ============================================================

def vertical_flip(image, objects):

    image = cv2.flip(
        image,
        0
    )

    new_objects = []

    for obj in objects:

        cid, xc, yc, w, h = obj

        yc = 1.0 - yc

        new_objects.append(
            [cid, xc, yc, w, h]
        )

    return image, new_objects


# ============================================================
# BOTH FLIPS
# ============================================================

def both_flip(image, objects):

    image = cv2.flip(
        image,
        -1
    )

    new_objects = []

    for obj in objects:

        cid, xc, yc, w, h = obj

        xc = 1.0 - xc
        yc = 1.0 - yc

        new_objects.append(
            [cid, xc, yc, w, h]
        )

    return image, new_objects


# ============================================================
# BRIGHTNESS / CONTRAST
# ============================================================

def brightness_contrast(image):

    alpha = random.uniform(
        0.85,
        1.15
    )

    beta = random.randint(
        -20,
        20
    )

    return cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta
    )


# ============================================================
# GAUSSIAN BLUR
# ============================================================

def mild_blur(image):

    kernel = random.choice(
        [
            (3, 3),
            (5, 5)
        ]
    )

    return cv2.GaussianBlur(
        image,
        kernel,
        0
    )


# ============================================================
# MILD IMAGE NOISE
# ============================================================

def add_noise(image):

    noise_strength = random.uniform(
        1.0,
        4.0
    )

    noise = np.random.normal(
        0,
        noise_strength,
        image.shape
    )

    noisy = (
        image.astype(
            np.float32
        )
        + noise
    )

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(
        np.uint8
    )

    return noisy


# ============================================================
# COMBINED AUGMENTATION
# ============================================================

def augment(image, objects):

    # --------------------------------------------------------
    # Random flip
    # --------------------------------------------------------

    flip_type = random.choice(
        [
            "none",
            "horizontal",
            "vertical",
            "both",
        ]
    )

    if flip_type == "horizontal":

        image, objects = horizontal_flip(
            image,
            objects
        )

    elif flip_type == "vertical":

        image, objects = vertical_flip(
            image,
            objects
        )

    elif flip_type == "both":

        image, objects = both_flip(
            image,
            objects
        )

    # --------------------------------------------------------
    # Photometric augmentation
    # --------------------------------------------------------

    if random.random() < 0.70:

        image = brightness_contrast(
            image
        )

    # --------------------------------------------------------
    # Mild blur
    # --------------------------------------------------------

    if random.random() < 0.15:

        image = mild_blur(
            image
        )

    # --------------------------------------------------------
    # Mild noise
    # --------------------------------------------------------

    if random.random() < 0.15:

        image = add_noise(
            image
        )

    return image, objects


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("FINAL RBC TARGETED AUGMENTATION")
    print("=" * 75)

    print(
        "\nTarget additional images:",
        TARGET_AUGMENTATIONS
    )

    print(
        "Dataset:",
        DATASET
    )

    print(
        "\nOnly TRAIN will be modified."
    )

    print(
        "VALIDATION and TEST will remain unchanged."
    )

    # --------------------------------------------------------
    # Find training images containing RBC
    # --------------------------------------------------------

    candidates = []

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    print(
        "\nScanning training labels..."
    )

    label_files = list(
        TRAIN_LABELS.glob("*.txt")
    )

    for label_path in label_files:

        objects = read_label(
            label_path
        )

        if not objects:
            continue

        has_rbc = any(
            obj[0] == RBC_CLASS
            for obj in objects
        )

        if not has_rbc:
            continue

        stem = label_path.stem

        image_path = None

        for ext in image_extensions:

            candidate = (
                TRAIN_IMAGES
                / f"{stem}{ext}"
            )

            if candidate.exists():

                image_path = candidate
                break

        if image_path is None:
            continue

        candidates.append(
            (
                image_path,
                label_path
            )
        )

    print(
        "RBC-containing training images:",
        len(candidates)
    )

    if len(candidates) == 0:

        print(
            "\nERROR: No RBC training images found."
        )

        return

    # --------------------------------------------------------
    # Shuffle source images
    # --------------------------------------------------------

    random.shuffle(
        candidates
    )

    # --------------------------------------------------------
    # Generate augmentations
    # --------------------------------------------------------

    created = 0
    failed = 0

    # Start after current largest augmentation index
    existing_augmented = []

    for p in TRAIN_IMAGES.glob(
        "rbc_final_aug_*"
    ):

        existing_augmented.append(p)

    start_index = (
        len(existing_augmented)
    )

    print(
        "\nStarting augmentation..."
    )

    while created < TARGET_AUGMENTATIONS:

        # ----------------------------------------------------
        # Select source
        # ----------------------------------------------------

        image_path, label_path = random.choice(
            candidates
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            failed += 1
            continue

        objects = read_label(
            label_path
        )

        if not objects:

            failed += 1
            continue

        # ----------------------------------------------------
        # Apply augmentation
        # ----------------------------------------------------

        aug_image, aug_objects = augment(
            image.copy(),
            [obj.copy() for obj in objects]
        )

        # ----------------------------------------------------
        # Output name
        # ----------------------------------------------------

        index = (
            start_index
            + created
            + 1
        )

        image_name = (
            f"rbc_final_aug_{index:05d}.jpg"
        )

        label_name = (
            f"rbc_final_aug_{index:05d}.txt"
        )

        output_image = (
            TRAIN_IMAGES
            / image_name
        )

        output_label = (
            TRAIN_LABELS
            / label_name
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        success = cv2.imwrite(
            str(output_image),
            aug_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        if not success:

            failed += 1
            continue

        write_label(
            output_label,
            aug_objects
        )

        created += 1

        if (
            created % 100 == 0
            or created == TARGET_AUGMENTATIONS
        ):

            print(
                f"Created: "
                f"{created:,} / "
                f"{TARGET_AUGMENTATIONS:,}"
            )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("RBC AUGMENTATION COMPLETE")
    print("=" * 75)

    print(
        f"New RBC augmented images : {created:,}"
    )

    print(
        f"Failures                  : {failed:,}"
    )

    print(
        "\nValidation images modified : 0"
    )

    print(
        "Test images modified       : 0"
    )

    print(
        "\nTraining directory:"
    )

    print(
        TRAIN_IMAGES
    )

    print("\nNext step:")
    print(
        "Vahadane stain normalization"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()