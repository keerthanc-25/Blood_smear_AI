from pathlib import Path
import shutil
import csv

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    r"D:\Blood_Smear_AI\runs\blood_smear_rbc_wbc_platelet_finetuned\weights\best.pt"
)

INPUT_DIR = Path(
    r"D:\Blood_Smear_AI\data\unannotated"
)

OUTPUT_DIR = Path(
    r"D:\Blood_Smear_AI\data\pseudo_labeled_3class"
)

IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"

SUMMARY_FILE = OUTPUT_DIR / "pseudo_label_summary.csv"


# Class IDs
# 0 = RBC
# 1 = WBC
# 2 = Platelet

CLASS_NAMES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet",
}


# Class-specific confidence thresholds
CONF_THRESHOLDS = {
    0: 0.90,   # RBC
    1: 0.85,   # WBC
    2: 0.85,   # Platelet
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Input folder not found:\n{INPUT_DIR}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("AUTOMATIC BLOOD SMEAR PSEUDO-LABELING")
print("=" * 70)

print("\nLoading model...")
print(MODEL_PATH)

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")


# ============================================================
# FIND IMAGES
# ============================================================

images = sorted(
    [
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)

print(f"\nInput images found: {len(images)}")

if len(images) == 0:
    print("No images found.")
    raise SystemExit(0)


# ============================================================
# CSV SUMMARY
# ============================================================

csv_rows = []

total_images = 0
accepted_images = 0
rejected_images = 0

total_rbc = 0
total_wbc = 0
total_platelet = 0


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, image_path in enumerate(images, start=1):

    total_images += 1

    print(
        f"\n[{index}/{len(images)}] "
        f"Processing: {image_path.name}"
    )

    try:

        results = model.predict(
            source=str(image_path),
            imgsz=800,
            conf=0.25,
            iou=0.45,
            device=0,
            verbose=False,
        )

        result = results[0]

        accepted_boxes = []

        image_width = result.orig_shape[1]
        image_height = result.orig_shape[0]

        # ----------------------------------------------------
        # PROCESS PREDICTIONS
        # ----------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                # Only RBC/WBC/Platelet
                if class_id not in CLASS_NAMES:
                    continue

                threshold = CONF_THRESHOLDS[class_id]

                # Class-specific confidence filtering
                if confidence < threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # ------------------------------------------------
                # Convert XYXY -> YOLO normalized format
                # ------------------------------------------------

                x_center = ((x1 + x2) / 2) / image_width
                y_center = ((y1 + y2) / 2) / image_height

                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height

                # Safety clipping
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))

                accepted_boxes.append(
                    (
                        class_id,
                        x_center,
                        y_center,
                        width,
                        height,
                        confidence,
                    )
                )

        # ----------------------------------------------------
        # NO HIGH-CONFIDENCE DETECTIONS
        # ----------------------------------------------------

        if not accepted_boxes:

            rejected_images += 1

            csv_rows.append(
                [
                    image_path.name,
                    "REJECTED",
                    0,
                    0,
                    0,
                    0,
                ]
            )

            print("  -> No high-confidence detections")

            continue

        # ----------------------------------------------------
        # OUTPUT FILES
        # ----------------------------------------------------

        output_image = IMAGE_DIR / image_path.name
        output_label = LABEL_DIR / f"{image_path.stem}.txt"

        shutil.copy2(image_path, output_image)

        # ----------------------------------------------------
        # WRITE YOLO LABEL
        # ----------------------------------------------------

        with open(
            output_label,
            "w",
            encoding="utf-8"
        ) as f:

            for (
                class_id,
                x_center,
                y_center,
                width,
                height,
                confidence,
            ) in accepted_boxes:

                f.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

        # ----------------------------------------------------
        # COUNT CLASSES
        # ----------------------------------------------------

        image_rbc = 0
        image_wbc = 0
        image_platelet = 0

        for box in accepted_boxes:

            class_id = box[0]

            if class_id == 0:
                image_rbc += 1
                total_rbc += 1

            elif class_id == 1:
                image_wbc += 1
                total_wbc += 1

            elif class_id == 2:
                image_platelet += 1
                total_platelet += 1

        accepted_images += 1

        # ----------------------------------------------------
        # SAVE CSV INFORMATION
        # ----------------------------------------------------

        csv_rows.append(
            [
                image_path.name,
                "ACCEPTED",
                len(accepted_boxes),
                image_rbc,
                image_wbc,
                image_platelet,
            ]
        )

        print(
            f"  -> Accepted: "
            f"{len(accepted_boxes)} objects "
            f"(RBC={image_rbc}, "
            f"WBC={image_wbc}, "
            f"Platelet={image_platelet})"
        )


    except Exception as e:

        rejected_images += 1

        print(
            f"  -> ERROR: {e}"
        )

        csv_rows.append(
            [
                image_path.name,
                "ERROR",
                0,
                0,
                0,
                0,
            ]
        )


# ============================================================
# WRITE CSV
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "image",
            "status",
            "total_objects",
            "RBC",
            "WBC",
            "Platelet",
        ]
    )

    writer.writerows(csv_rows)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PSEUDO-LABELING COMPLETE")
print("=" * 70)

print(f"\nTotal input images       : {total_images}")
print(f"Accepted images          : {accepted_images}")
print(f"Rejected images          : {rejected_images}")

print("\nPseudo-label objects:")
print(f"RBC                      : {total_rbc}")
print(f"WBC                      : {total_wbc}")
print(f"Platelet                 : {total_platelet}")
print(
    f"TOTAL                    : "
    f"{total_rbc + total_wbc + total_platelet}"
)

print("\nOutput:")
print(f"Images                   : {IMAGE_DIR}")
print(f"Labels                   : {LABEL_DIR}")
print(f"Summary CSV              : {SUMMARY_FILE}")

print("\nConfidence thresholds:")
print("RBC                      : >= 0.90")
print("WBC                      : >= 0.85")
print("Platelet                 : >= 0.85")

print("\n" + "=" * 70)