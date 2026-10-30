from pathlib import Path
import hashlib
import shutil
import random
import csv

from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path(r"D:\blood smear images")

VERIFIED_DATASET = Path(
    r"D:\Blood_Smear_AI\data\yolo_dataset_3class"
)

OUTPUT_DIR = Path(
    r"D:\Blood_Smear_AI\data\pseudo_labeled_new"
)

MODEL_PATH = Path(
    r"D:\Blood_Smear_AI\runs\blood_smear_rbc_wbc_platelet_finetuned"
    r"\weights\best.pt"
)

# Process 75% of the TRUE unannotated pool
PROCESS_PERCENTAGE = 0.75

# Split the selected 75% into two equal groups
RBC_PERCENTAGE = 0.50
WBC_PLATELET_PERCENTAGE = 0.50

# User requested: strictly ABOVE 0.50
CONF_THRESHOLD = 0.50

# YOLO inference settings
IMGSZ = 800
BATCH_SIZE = 2
DEVICE = 0

# Reproducible selection
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

CLASS_NAMES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet"
}


# ============================================================
# HASH
# ============================================================

def get_hash(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# IMAGE VALIDATION
# ============================================================

def is_valid_image(path):
    """
    Check that the image can actually be opened and decoded.

    Invalid/corrupt images are skipped instead of stopping the
    entire 75% annotation run.
    """
    try:
        with Image.open(path) as img:
            img.verify()

        # verify() invalidates the object, so reopen it
        with Image.open(path) as img:
            img.load()

        return True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError
    ):
        return False


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(folder):
    images = []

    if not folder.exists():
        print(f"[ERROR] Folder does not exist: {folder}")
        return images

    for p in folder.rglob("*"):

        if not p.is_file():
            continue

        # Ignore macOS metadata folders
        if "_MACOSX" in p.parts:
            continue

        if p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)

    return images


# ============================================================
# CHECK WHETHER IMAGE HAS A SIDE-CAR ANNOTATION
# ============================================================

def has_sidecar_annotation(image_path):

    stem = image_path.stem
    parent = image_path.parent

    possible_extensions = [
        ".txt",
        ".xml",
        ".json"
    ]

    for ext in possible_extensions:

        candidate = parent / f"{stem}{ext}"

        if candidate.exists() and candidate.stat().st_size > 0:
            return True

    return False


# ============================================================
# COLLECT HASHES FROM VERIFIED DATASET
# ============================================================

def collect_verified_hashes():

    print("\nCollecting hashes from verified dataset...")

    verified_images = find_images(
        VERIFIED_DATASET / "images"
    )

    hashes = set()

    for i, image in enumerate(verified_images, 1):

        try:
            hashes.add(get_hash(image))

        except Exception as e:
            print(f"[WARNING] Hash error: {image} -> {e}")

        if i % 500 == 0:
            print(
                f"  Hashed {i}/{len(verified_images)} "
                f"verified images..."
            )

    print(
        f"Verified dataset images found: "
        f"{len(verified_images)}"
    )

    print(
        f"Unique verified image hashes: "
        f"{len(hashes)}"
    )

    return hashes


# ============================================================
# FIND TRUE UNANNOTATED IMAGES
# ============================================================

def collect_unannotated_images(verified_hashes):

    print("\nScanning original source...")
    print(SOURCE_DIR)

    all_images = find_images(SOURCE_DIR)

    print(
        f"\nTotal valid image files found before validation: "
        f"{len(all_images)}"
    )

    unannotated = []

    sidecar_annotated = 0
    duplicate_verified = 0
    duplicate_source = 0
    invalid_images = 0

    seen_hashes = set()

    for i, image in enumerate(all_images, 1):

        # ----------------------------------------------------
        # Validate image before hashing / prediction
        # ----------------------------------------------------

        if not is_valid_image(image):

            invalid_images += 1

            if invalid_images <= 20:
                print(
                    f"[SKIP INVALID/CORRUPT] {image}"
                )

            continue

        # ----------------------------------------------------
        # Hash image
        # ----------------------------------------------------

        try:
            image_hash = get_hash(image)

        except Exception as e:

            invalid_images += 1

            if invalid_images <= 20:
                print(
                    f"[SKIP HASH ERROR] {image} -> {e}"
                )

            continue

        # ----------------------------------------------------
        # Remove duplicate copies in source
        # ----------------------------------------------------

        if image_hash in seen_hashes:

            duplicate_source += 1
            continue

        seen_hashes.add(image_hash)

        # ----------------------------------------------------
        # Already present in verified YOLO dataset
        # ----------------------------------------------------

        if image_hash in verified_hashes:

            duplicate_verified += 1
            continue

        # ----------------------------------------------------
        # Existing sidecar annotation
        # ----------------------------------------------------

        if has_sidecar_annotation(image):

            sidecar_annotated += 1
            continue

        # ----------------------------------------------------
        # TRUE UNANNOTATED IMAGE
        # ----------------------------------------------------

        unannotated.append(image)

        if i % 1000 == 0:

            print(
                f"Scanned {i}/{len(all_images)} | "
                f"true unannotated: {len(unannotated)}"
            )

    print("\n" + "=" * 70)
    print("SOURCE SUMMARY")
    print("=" * 70)

    print(
        f"Image files scanned:             {len(all_images)}"
    )

    print(
        f"Invalid/corrupt skipped:         {invalid_images}"
    )

    print(
        f"Duplicates inside source:        {duplicate_source}"
    )

    print(
        f"Already in verified dataset:     {duplicate_verified}"
    )

    print(
        f"Sidecar annotated:                {sidecar_annotated}"
    )

    print(
        f"TRUE UNANNOTATED:                 {len(unannotated)}"
    )

    return unannotated, invalid_images


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_output_dirs():

    folders = [
        OUTPUT_DIR / "rbc" / "images",
        OUTPUT_DIR / "rbc" / "labels",
        OUTPUT_DIR / "wbc_platelet" / "images",
        OUTPUT_DIR / "wbc_platelet" / "labels"
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAFE OUTPUT NAME
# ============================================================

def output_name(image_path):

    h = get_hash(image_path)[:12]

    return (
        f"{image_path.stem}_"
        f"{h}"
        f"{image_path.suffix.lower()}"
    )


# ============================================================
# WRITE YOLO LABEL
# ============================================================

def save_label(label_path, detections):

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as f:

        for cls_id, x, y, w, h, conf in detections:

            # YOLO label format:
            # class x_center y_center width height
            f.write(
                f"{cls_id} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


# ============================================================
# PROCESS GROUP
# ============================================================

def process_group(
    model,
    images,
    group_name
):

    if group_name == "rbc":

        output_image_dir = (
            OUTPUT_DIR / "rbc" / "images"
        )

        output_label_dir = (
            OUTPUT_DIR / "rbc" / "labels"
        )

        # RBC-focused group:
        # only RBC detections are written.
        allowed_classes = {0}

    else:

        output_image_dir = (
            OUTPUT_DIR / "wbc_platelet" / "images"
        )

        output_label_dir = (
            OUTPUT_DIR / "wbc_platelet" / "labels"
        )

        # WBC + Platelet group:
        # only WBC and Platelet detections are written.
        allowed_classes = {1, 2}

    print("\n")
    print("=" * 70)
    print(f"PROCESSING GROUP: {group_name.upper()}")
    print("=" * 70)

    print(f"Images assigned: {len(images)}")
    print(f"Confidence requirement: > {CONF_THRESHOLD}")

    accepted_images = 0
    prediction_errors = 0

    class_counts = {
        0: 0,
        1: 0,
        2: 0
    }

    csv_rows = []

    # --------------------------------------------------------
    # Process in small batches for GTX 1050 Ti
    # --------------------------------------------------------

    for start in range(0, len(images), BATCH_SIZE):

        batch = images[start:start + BATCH_SIZE]

        print(
            f"\nProcessing "
            f"{start + 1}-"
            f"{min(start + BATCH_SIZE, len(images))}"
            f"/{len(images)}"
        )

        # ----------------------------------------------------
        # Protect the whole batch from prediction failure
        # ----------------------------------------------------

        try:

            results = model.predict(
                source=[str(x) for x in batch],
                conf=CONF_THRESHOLD,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False
            )

        except Exception as e:

            prediction_errors += len(batch)

            print(
                "\n[WARNING] Batch prediction failed."
            )

            print(
                f"Reason: {e}"
            )

            # Try each image separately so one bad image
            # does not discard the entire batch.
            results = []

            for image_path in batch:

                try:

                    single_result = model.predict(
                        source=str(image_path),
                        conf=CONF_THRESHOLD,
                        imgsz=IMGSZ,
                        device=DEVICE,
                        verbose=False
                    )

                    if single_result:
                        results.append(single_result[0])
                    else:
                        results.append(None)

                except Exception as single_error:

                    prediction_errors += 1

                    print(
                        f"[SKIP PREDICTION ERROR] "
                        f"{image_path}"
                    )

                    print(
                        f"  {single_error}"
                    )

                    results.append(None)

        # ----------------------------------------------------
        # Process results
        # ----------------------------------------------------

        for image_path, result in zip(batch, results):

            if result is None:

                csv_rows.append([
                    str(image_path),
                    group_name,
                    0,
                    0,
                    0,
                    "PREDICTION_ERROR",
                    0,
                    0,
                    0
                ])

                continue

            detections = []

            if result.boxes is None:

                csv_rows.append([
                    str(image_path),
                    group_name,
                    0,
                    0,
                    0,
                    "REJECTED",
                    0,
                    0,
                    0
                ])

                continue

            # ------------------------------------------------
            # Extract detections
            # ------------------------------------------------

            for box in result.boxes:

                cls_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                # STRICTLY ABOVE 0.50
                if confidence <= CONF_THRESHOLD:
                    continue

                # Keep only the classes assigned to this group
                if cls_id not in allowed_classes:
                    continue

                x, y, w, h = (
                    box.xywhn[0].tolist()
                )

                detections.append(
                    (
                        cls_id,
                        x,
                        y,
                        w,
                        h,
                        confidence
                    )
                )

                class_counts[cls_id] += 1

            # ------------------------------------------------
            # No qualifying detection
            # ------------------------------------------------

            if not detections:

                csv_rows.append([
                    str(image_path),
                    group_name,
                    0,
                    0,
                    0,
                    "REJECTED",
                    0,
                    0,
                    0
                ])

                continue

            # ------------------------------------------------
            # Copy image
            # ------------------------------------------------

            try:

                new_name = output_name(
                    image_path
                )

                destination_image = (
                    output_image_dir / new_name
                )

                destination_label = (
                    output_label_dir /
                    f"{Path(new_name).stem}.txt"
                )

                shutil.copy2(
                    image_path,
                    destination_image
                )

                save_label(
                    destination_label,
                    detections
                )

            except Exception as e:

                print(
                    f"[WARNING] Could not save "
                    f"{image_path}: {e}"
                )

                csv_rows.append([
                    str(image_path),
                    group_name,
                    0,
                    0,
                    0,
                    "SAVE_ERROR",
                    0,
                    0,
                    0
                ])

                continue

            accepted_images += 1

            # ------------------------------------------------
            # Counts
            # ------------------------------------------------

            rbc_count = sum(
                1
                for d in detections
                if d[0] == 0
            )

            wbc_count = sum(
                1
                for d in detections
                if d[0] == 1
            )

            platelet_count = sum(
                1
                for d in detections
                if d[0] == 2
            )

            max_rbc_conf = max(
                [
                    d[5]
                    for d in detections
                    if d[0] == 0
                ],
                default=0
            )

            max_wbc_conf = max(
                [
                    d[5]
                    for d in detections
                    if d[0] == 1
                ],
                default=0
            )

            max_platelet_conf = max(
                [
                    d[5]
                    for d in detections
                    if d[0] == 2
                ],
                default=0
            )

            csv_rows.append([
                str(image_path),
                group_name,
                rbc_count,
                wbc_count,
                platelet_count,
                "ACCEPTED",
                f"{max_rbc_conf:.4f}",
                f"{max_wbc_conf:.4f}",
                f"{max_platelet_conf:.4f}"
            ])

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        processed = min(
            start + BATCH_SIZE,
            len(images)
        )

        print(
            f"Progress: {processed}/{len(images)}"
        )

        print(
            f"Accepted: {accepted_images}"
        )

        print(
            f"RBC objects: {class_counts[0]} | "
            f"WBC objects: {class_counts[1]} | "
            f"Platelet objects: {class_counts[2]}"
        )

    return (
        csv_rows,
        accepted_images,
        class_counts,
        prediction_errors
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

def save_summary(rows):

    csv_path = (
        OUTPUT_DIR /
        "pseudo_label_summary.csv"
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
            "group",
            "RBC_count",
            "WBC_count",
            "Platelet_count",
            "status",
            "max_RBC_conf",
            "max_WBC_conf",
            "max_Platelet_conf"
        ])

        writer.writerows(rows)

    return csv_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FRESH BLOOD SMEAR RE-ANNOTATION")
    print("=" * 70)

    print("\nSource:")
    print(SOURCE_DIR)

    print("\nVerified dataset:")
    print(VERIFIED_DATASET)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nConfidence:")
    print("> 0.50 for every retained detection")

    print("\nProcessing:")
    print("75% of TRUE unannotated images")

    print("\nSplit:")
    print("50% RBC-focused")
    print("50% WBC + Platelet-focused")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # VERIFIED HASHES
    # --------------------------------------------------------

    verified_hashes = (
        collect_verified_hashes()
    )

    # --------------------------------------------------------
    # FIND TRUE UNANNOTATED
    # --------------------------------------------------------

    (
        unannotated,
        invalid_images
    ) = collect_unannotated_images(
        verified_hashes
    )

    if not unannotated:

        print(
            "\nNo unannotated images found."
        )

        return

    # --------------------------------------------------------
    # DETERMINISTIC SHUFFLE
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    random.shuffle(unannotated)

    # --------------------------------------------------------
    # SELECT 75%
    # --------------------------------------------------------

    target_count = int(
        len(unannotated) *
        PROCESS_PERCENTAGE
    )

    selected = (
        unannotated[:target_count]
    )

    remaining = (
        unannotated[target_count:]
    )

    # --------------------------------------------------------
    # SPLIT 50 / 50
    # --------------------------------------------------------

    rbc_count = int(
        len(selected) *
        RBC_PERCENTAGE
    )

    rbc_images = (
        selected[:rbc_count]
    )

    wbc_platelet_images = (
        selected[rbc_count:]
    )

    # --------------------------------------------------------
    # CREATE OUTPUT
    # --------------------------------------------------------

    create_output_dirs()

    # --------------------------------------------------------
    # PRINT PLAN
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ANNOTATION PLAN")
    print("=" * 70)

    print(
        f"Total TRUE unannotated: "
        f"{len(unannotated)}"
    )

    print(
        f"Target 75%: "
        f"{target_count}"
    )

    print(
        f"RBC group: "
        f"{len(rbc_images)}"
    )

    print(
        f"WBC + Platelet group: "
        f"{len(wbc_platelet_images)}"
    )

    print(
        f"Remaining untouched: "
        f"{len(remaining)}"
    )

    print(
        f"Invalid/corrupt skipped: "
        f"{invalid_images}"
    )

    # --------------------------------------------------------
    # PROCESS RBC GROUP
    # --------------------------------------------------------

    (
        rbc_rows,
        rbc_accepted,
        rbc_classes,
        rbc_errors
    ) = process_group(
        model,
        rbc_images,
        "rbc"
    )

    # --------------------------------------------------------
    # PROCESS WBC + PLATELET GROUP
    # --------------------------------------------------------

    (
        wp_rows,
        wp_accepted,
        wp_classes,
        wp_errors
    ) = process_group(
        model,
        wbc_platelet_images,
        "wbc_platelet"
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    all_rows = (
        rbc_rows +
        wp_rows
    )

    csv_path = save_summary(
        all_rows
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL RE-ANNOTATION REPORT")
    print("=" * 70)

    print(
        f"\nTotal TRUE unannotated: "
        f"{len(unannotated)}"
    )

    print(
        f"Selected for 75% processing: "
        f"{target_count}"
    )

    print(
        f"Remaining untouched: "
        f"{len(remaining)}"
    )

    print(
        f"Invalid/corrupt skipped: "
        f"{invalid_images}"
    )

    print("\nRBC GROUP")

    print(
        f"Images processed: "
        f"{len(rbc_images)}"
    )

    print(
        f"Images accepted: "
        f"{rbc_accepted}"
    )

    print(
        f"RBC objects: "
        f"{rbc_classes[0]}"
    )

    print(
        f"Prediction errors: "
        f"{rbc_errors}"
    )

    print("\nWBC + PLATELET GROUP")

    print(
        f"Images processed: "
        f"{len(wbc_platelet_images)}"
    )

    print(
        f"Images accepted: "
        f"{wp_accepted}"
    )

    print(
        f"WBC objects: "
        f"{wp_classes[1]}"
    )

    print(
        f"Platelet objects: "
        f"{wp_classes[2]}"
    )

    print(
        f"Prediction errors: "
        f"{wp_errors}"
    )

    print("\nOUTPUT:")
    print(OUTPUT_DIR)

    print("\nCSV:")
    print(csv_path)

    print("\nDONE.")


if __name__ == "__main__":
    main()
