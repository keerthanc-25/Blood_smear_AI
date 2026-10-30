import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\Blood_Smear_AI")

MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "blood_smear_rbc_wbc_platelet"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = (
    PROJECT_DIR
    / "data"
    / "yolo_dataset_3class"
    / "images"
    / "test"
)

LABEL_DIR = (
    PROJECT_DIR
    / "data"
    / "yolo_dataset_3class"
    / "labels"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "runs"
    / "rbc_wbc_comparison"
)

CSV_PATH = OUTPUT_DIR / "rbc_wbc_errors.csv"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 800
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50

CLASS_NAMES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet"
}


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# LOAD YOLO GROUND TRUTH
# ============================================================

def load_ground_truth(label_path, width, height):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height

        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2

        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        boxes.append({
            "class_id": class_id,
            "box": [x1, y1, x2, y2]
        })

    return boxes


# ============================================================
# DRAW BOX
# ============================================================

def draw_box(
    image,
    box,
    label,
    color,
    thickness=2
):

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    cv2.putText(
        image,
        label,
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RBC / WBC GROUND-TRUTH VS PREDICTION ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading model...")

    model = YOLO(str(MODEL_PATH))

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    image_files = sorted(
        [
            p for p in IMAGE_DIR.iterdir()
            if p.suffix.lower() in image_extensions
        ]
    )

    print("Test images:", len(image_files))

    error_images = 0
    error_objects = 0

    csv_rows = []

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        # ----------------------------------------------------
        # Read original image
        # ----------------------------------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image_height, image_width = image.shape[:2]

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        label_path = (
            LABEL_DIR /
            f"{image_path.stem}.txt"
        )

        ground_truth = load_ground_truth(
            label_path,
            image_width,
            image_height
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=0.45,
            device=0,
            verbose=False
        )

        result = results[0]

        predictions = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            for box, class_id, confidence in zip(
                boxes,
                classes,
                confidences
            ):

                predictions.append({
                    "class_id": int(class_id),
                    "box": box.tolist(),
                    "confidence": float(confidence)
                })

        # ----------------------------------------------------
        # Find RBC -> WBC errors
        # ----------------------------------------------------

        image_errors = []

        for gt_index, gt in enumerate(ground_truth):

            # Only actual RBC
            if gt["class_id"] != 0:
                continue

            best_iou = 0.0
            best_prediction = None

            for pred_index, prediction in enumerate(
                predictions
            ):

                iou = calculate_iou(
                    gt["box"],
                    prediction["box"]
                )

                if iou > best_iou:

                    best_iou = iou
                    best_prediction = prediction

            # Actual RBC matched to predicted WBC
            if (
                best_prediction is not None
                and best_prediction["class_id"] == 1
                and best_iou >= IOU_THRESHOLD
            ):

                image_errors.append({
                    "gt_box": gt["box"],
                    "pred_box": best_prediction["box"],
                    "iou": best_iou,
                    "confidence": best_prediction["confidence"]
                })

        # ----------------------------------------------------
        # No RBC -> WBC error
        # ----------------------------------------------------

        if len(image_errors) == 0:

            if index % 10 == 0:
                print(
                    f"Processed {index}/{len(image_files)}"
                )

            continue

        # ----------------------------------------------------
        # Draw comparison
        # ----------------------------------------------------

        comparison = image.copy()

        # Ground truth boxes
        #
        # Green = Ground Truth
        #

        for gt in ground_truth:

            class_name = CLASS_NAMES.get(
                gt["class_id"],
                "Unknown"
            )

            draw_box(
                comparison,
                gt["box"],
                f"GT {class_name}",
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # Draw ALL predictions
        #
        # Blue = normal predictions
        # Red = WBC prediction overlapping actual RBC
        # ----------------------------------------------------

        for prediction in predictions:

            class_name = CLASS_NAMES.get(
                prediction["class_id"],
                "Unknown"
            )

            color = (255, 0, 0)

            draw_box(
                comparison,
                prediction["box"],
                f"PRED {class_name} "
                f"{prediction['confidence']:.2f}",
                color,
                2
            )

        # ----------------------------------------------------
        # Highlight RBC -> WBC errors
        # ----------------------------------------------------

        for error in image_errors:

            draw_box(
                comparison,
                error["gt_box"],
                "ERROR: GT RBC",
                (0, 0, 255),
                4
            )

            draw_box(
                comparison,
                error["pred_box"],
                (
                    f"ERROR: PRED WBC "
                    f"{error['confidence']:.2f}"
                ),
                (0, 0, 255),
                4
            )

        # ----------------------------------------------------
        # Add title
        # ----------------------------------------------------

        title = (
            f"RBC -> WBC errors: "
            f"{len(image_errors)}"
        )

        cv2.putText(
            comparison,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Save ONLY ONCE per image
        # ----------------------------------------------------

        output_filename = (
            f"error_{error_images + 1:04d}_"
            f"{image_path.name}"
        )

        output_path = (
            OUTPUT_DIR /
            output_filename
        )

        cv2.imwrite(
            str(output_path),
            comparison
        )

        # ----------------------------------------------------
        # CSV records
        # ----------------------------------------------------

        for error in image_errors:

            csv_rows.append([
                image_path.name,
                error["iou"],
                error["confidence"]
            ])

        error_objects += len(image_errors)
        error_images += 1

        print(
            f"Image {index}/{len(image_files)} | "
            f"{image_path.name} | "
            f"RBC -> WBC errors: "
            f"{len(image_errors)}"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "image",
            "IoU",
            "prediction_confidence"
        ])

        writer.writerows(csv_rows)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nImages containing RBC -> WBC errors: "
        f"{error_images}"
    )

    print(
        f"Total RBC -> WBC error objects: "
        f"{error_objects}"
    )

    print("\nComparison images saved to:")
    print(OUTPUT_DIR)

    print("\nCSV saved to:")
    print(CSV_PATH)


if __name__ == "__main__":
    main()