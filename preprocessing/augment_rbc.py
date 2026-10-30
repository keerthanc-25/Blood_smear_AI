from pathlib import Path
import cv2
import random
import shutil
import math

# ============================================================
# PATHS
# ============================================================

SOURCE = Path(
    r"D:\Blood_Smear_AI\data\yolo_dataset_3class"
)

OUTPUT = Path(
    r"D:\Blood_Smear_AI\data\rbc_augmented"
)

IMAGE_DIR = SOURCE / "images" / "train"
LABEL_DIR = SOURCE / "labels" / "train"

OUTPUT_IMAGE_DIR = OUTPUT / "images"
OUTPUT_LABEL_DIR = OUTPUT / "labels"

# ============================================================
# SETTINGS
# ============================================================

RBC_CLASS = 0

TARGET_AUGMENTED_IMAGES = 3000

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

random.seed(42)

# ============================================================
# RECREATE OUTPUT FOLDER
# ============================================================

if OUTPUT.exists():
    print("Removing previous augmentation folder...")
    shutil.rmtree(OUTPUT)

OUTPUT_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# FIND RBC-CONTAINING IMAGES
# ============================================================

images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
])

rbc_sources = []

total_images = 0
valid_pairs = 0
invalid_pairs = 0
rbc_objects = 0

invalid_details = []

for image_path in images:

    total_images += 1

    label_path = LABEL_DIR / (
        image_path.stem + ".txt"
    )

    if not label_path.exists():

        invalid_pairs += 1
        invalid_details.append(
            f"{image_path.name}: missing label"
        )
        continue

    image = cv2.imread(str(image_path))

    if image is None:

        invalid_pairs += 1
        invalid_details.append(
            f"{image_path.name}: unreadable image"
        )
        continue

    annotations = []

    try:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                annotations.append(
                    [cls, x, y, w, h]
                )

    except Exception as e:

        invalid_pairs += 1
        invalid_details.append(
            f"{image_path.name}: {e}"
        )
        continue

    valid_pairs += 1

    rbc_count = sum(
        1
        for ann in annotations
        if ann[0] == RBC_CLASS
    )

    if rbc_count > 0:

        rbc_sources.append({
            "image": image_path,
            "annotations": annotations,
            "rbc_count": rbc_count
        })

        rbc_objects += rbc_count

# ============================================================
# INITIAL REPORT
# ============================================================

print("=" * 70)
print("RBC AUGMENTATION")
print("=" * 70)

print(f"Total training images      : {total_images}")
print(f"Valid image/label pairs    : {valid_pairs}")
print(f"Invalid pairs              : {invalid_pairs}")
print(f"RBC-containing images      : {len(rbc_sources)}")
print(f"RBC objects                : {rbc_objects}")
print(f"Target augmented images    : {TARGET_AUGMENTED_IMAGES}")
print("=" * 70)

if not rbc_sources:
    raise RuntimeError(
        "No RBC-containing images found."
    )

# ============================================================
# AUGMENTATION FUNCTION
# ============================================================

def rotate_image_and_boxes(
    image,
    annotations,
    angle
):
    """
    Rotate image with expanded canvas and transform
    YOLO bounding boxes using corner coordinates.
    """

    h, w = image.shape[:2]

    center = (
        w / 2,
        h / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int(
        h * sin + w * cos
    )

    new_h = int(
        h * cos + w * sin
    )

    matrix[0, 2] += (
        new_w / 2 - center[0]
    )

    matrix[1, 2] += (
        new_h / 2 - center[1]
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_REFLECT_101
    )

    transformed = []

    for cls, x, y, bw, bh in annotations:

        # Convert YOLO normalized box to pixels
        cx = x * w
        cy = y * h

        box_w = bw * w
        box_h = bh * h

        x1 = cx - box_w / 2
        y1 = cy - box_h / 2
        x2 = cx + box_w / 2
        y2 = cy + box_h / 2

        corners = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]

        transformed_corners = []

        for px, py in corners:

            nx = (
                matrix[0, 0] * px
                + matrix[0, 1] * py
                + matrix[0, 2]
            )

            ny = (
                matrix[1, 0] * px
                + matrix[1, 1] * py
                + matrix[1, 2]
            )

            transformed_corners.append(
                [nx, ny]
            )

        xs = [
            p[0]
            for p in transformed_corners
        ]

        ys = [
            p[1]
            for p in transformed_corners
        ]

        nx1 = max(0, min(xs))
        ny1 = max(0, min(ys))
        nx2 = min(new_w, max(xs))
        ny2 = min(new_h, max(ys))

        if nx2 <= nx1 or ny2 <= ny1:
            continue

        new_cx = (
            (nx1 + nx2) / 2
        ) / new_w

        new_cy = (
            (ny1 + ny2) / 2
        ) / new_h

        new_bw = (
            nx2 - nx1
        ) / new_w

        new_bh = (
            ny2 - ny1
        ) / new_h

        transformed.append([
            cls,
            new_cx,
            new_cy,
            new_bw,
            new_bh
        ])

    return rotated, transformed


# ============================================================
# APPLY AUGMENTATIONS
# ============================================================

def augment_image(
    image,
    annotations
):

    result = image.copy()
    transformed = annotations.copy()

    h, w = result.shape[:2]

    # --------------------------------------------------------
    # Random rotation
    # --------------------------------------------------------

    rotation_type = random.choice([
        "small",
        "small",
        "small",
        "90",
        "180",
        "270",
        "none"
    ])

    if rotation_type == "small":

        angle = random.uniform(
            -25,
            25
        )

        result, transformed = rotate_image_and_boxes(
            result,
            transformed,
            angle
        )

    elif rotation_type == "90":

        result, transformed = rotate_image_and_boxes(
            result,
            transformed,
            90
        )

    elif rotation_type == "180":

        result, transformed = rotate_image_and_boxes(
            result,
            transformed,
            180
        )

    elif rotation_type == "270":

        result, transformed = rotate_image_and_boxes(
            result,
            transformed,
            270
        )

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    if random.random() < 0.5:

        result = cv2.flip(
            result,
            1
        )

        for ann in transformed:

            ann[1] = 1.0 - ann[1]

    # --------------------------------------------------------
    # Vertical flip
    # --------------------------------------------------------

    if random.random() < 0.25:

        result = cv2.flip(
            result,
            0
        )

        for ann in transformed:

            ann[2] = 1.0 - ann[2]

    # --------------------------------------------------------
    # Mild brightness / contrast
    # --------------------------------------------------------

    if random.random() < 0.35:

        alpha = random.uniform(
            0.90,
            1.10
        )

        beta = random.randint(
            -10,
            10
        )

        result = cv2.convertScaleAbs(
            result,
            alpha=alpha,
            beta=beta
        )

    # --------------------------------------------------------
    # Mild Gaussian blur
    # --------------------------------------------------------

    if random.random() < 0.10:

        result = cv2.GaussianBlur(
            result,
            (3, 3),
            0
        )

    return result, transformed


# ============================================================
# GENERATE EXACTLY 3000
# ============================================================

print()
print("Generating augmented images...")
print("=" * 70)

for index in range(
    TARGET_AUGMENTED_IMAGES
):

    # Randomly select one of the 739 RBC images
    source = random.choice(
        rbc_sources
    )

    image_path = source["image"]
    annotations = source["annotations"]

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        continue

    augmented, new_annotations = augment_image(
        image,
        annotations
    )

    if not new_annotations:
        continue

    # --------------------------------------------------------
    # Make unique filename
    # --------------------------------------------------------

    output_name = (
        f"rbc_aug_{index + 1:06d}_"
        f"{image_path.stem}.jpg"
    )

    output_image = (
        OUTPUT_IMAGE_DIR /
        output_name
    )

    output_label = (
        OUTPUT_LABEL_DIR /
        f"{Path(output_name).stem}.txt"
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(output_image),
        augmented,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not success:
        continue

    # --------------------------------------------------------
    # Save YOLO labels
    # --------------------------------------------------------

    with open(
        output_label,
        "w",
        encoding="utf-8"
    ) as f:

        for cls, x, y, bw, bh in new_annotations:

            # Final safety clamp
            x = max(
                0.0,
                min(1.0, x)
            )

            y = max(
                0.0,
                min(1.0, y)
            )

            bw = max(
                0.000001,
                min(1.0, bw)
            )

            bh = max(
                0.000001,
                min(1.0, bh)
            )

            f.write(
                f"{cls} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f}\n"
            )

    if (index + 1) % 250 == 0:

        print(
            f"Generated: "
            f"{index + 1}/{TARGET_AUGMENTED_IMAGES}"
        )

# ============================================================
# FINAL VERIFICATION
# ============================================================

generated_images = list(
    OUTPUT_IMAGE_DIR.glob("*.jpg")
)

generated_labels = list(
    OUTPUT_LABEL_DIR.glob("*.txt")
)

print()
print("=" * 70)
print("AUGMENTATION COMPLETE")
print("=" * 70)

print(
    f"Original RBC-containing images : "
    f"{len(rbc_sources)}"
)

print(
    f"Original RBC objects            : "
    f"{rbc_objects}"
)

print(
    f"Target augmented images         : "
    f"{TARGET_AUGMENTED_IMAGES}"
)

print(
    f"Generated augmented images      : "
    f"{len(generated_images)}"
)

print(
    f"Generated augmented labels      : "
    f"{len(generated_labels)}"
)

print(
    f"Image/label difference          : "
    f"{abs(len(generated_images) - len(generated_labels))}"
)

print()
print(
    f"Output folder: {OUTPUT}"
)

print("=" * 70)

# ============================================================
# INVALID REPORT
# ============================================================

if invalid_details:

    report = OUTPUT / "invalid_files.txt"

    with open(
        report,
        "w",
        encoding="utf-8"
    ) as f:

        for item in invalid_details:
            f.write(
                item + "\n"
            )

    print(
        f"Invalid report: {report}"
    )

print("=" * 70)