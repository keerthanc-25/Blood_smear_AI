import os
from pathlib import Path
from collections import Counter

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
# IOU FUNCTION
# ============================================================

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.

    Boxes are:
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(label_path, image_width, image_height):

    ground_truth = []

    if not label_path.exists():
        return ground_truth

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        ground_truth.append({
            "class_id": class_id,
            "box": [x1, y1, x2, y2]
        })

    return ground_truth


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RBC / WBC / PLATELET CONFUSION ANALYSIS")
    print("=" * 70)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nTest images:")
    print(IMAGE_DIR)

    print("\nIoU threshold:", IOU_THRESHOLD)
    print("Confidence threshold:", CONFIDENCE)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    confusion = Counter()

    missed_ground_truth = Counter()

    false_predictions = Counter()

    total_ground_truth = Counter()

    total_predictions = Counter()

    images_processed = 0

    # --------------------------------------------------------
    # Get images
    # --------------------------------------------------------

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    image_files = [
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    image_files.sort()

    print("\nImages found:", len(image_files))

    # --------------------------------------------------------
    # Process each image
    # --------------------------------------------------------

    for index, image_path in enumerate(image_files, start=1):

        # ----------------------------------------------------
        # Run prediction
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

        # ----------------------------------------------------
        # Image dimensions
        # ----------------------------------------------------

        image_height, image_width = result.orig_shape

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        gt_boxes = load_ground_truth(
            label_path,
            image_width,
            image_height
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

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
        # Count totals
        # ----------------------------------------------------

        for gt in gt_boxes:
            total_ground_truth[gt["class_id"]] += 1

        for pred in predictions:
            total_predictions[pred["class_id"]] += 1

        # ----------------------------------------------------
        # Match predictions to GT
        # ----------------------------------------------------

        matched_gt = set()
        matched_predictions = set()

        matches = []

        # Calculate all possible matches

        for gt_index, gt in enumerate(gt_boxes):

            for pred_index, pred in enumerate(predictions):

                iou = calculate_iou(
                    gt["box"],
                    pred["box"]
                )

                if iou >= IOU_THRESHOLD:

                    matches.append(
                        (
                            iou,
                            gt_index,
                            pred_index
                        )
                    )

        # Highest IoU matches first

        matches.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        for iou, gt_index, pred_index in matches:

            if gt_index in matched_gt:
                continue

            if pred_index in matched_predictions:
                continue

            matched_gt.add(gt_index)
            matched_predictions.add(pred_index)

            actual_class = gt_boxes[gt_index]["class_id"]
            predicted_class = predictions[pred_index]["class_id"]

            confusion[
                (
                    actual_class,
                    predicted_class
                )
            ] += 1

        # ----------------------------------------------------
        # Missed GT boxes
        # ----------------------------------------------------

        for gt_index, gt in enumerate(gt_boxes):

            if gt_index not in matched_gt:

                missed_ground_truth[
                    gt["class_id"]
                ] += 1

        # ----------------------------------------------------
        # False predictions
        # ----------------------------------------------------

        for pred_index, pred in enumerate(predictions):

            if pred_index not in matched_predictions:

                false_predictions[
                    pred["class_id"]
                ] += 1

        images_processed += 1

        if index % 10 == 0 or index == len(image_files):

            print(
                f"Processed {index}/{len(image_files)} images"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CONFUSION RESULTS")
    print("=" * 70)

    print(
        "\n                    Predicted"
    )

    print(
        "Actual          RBC        WBC        Platelet"
    )

    print("-" * 70)

    for actual_class in [0, 1, 2]:

        values = []

        for predicted_class in [0, 1, 2]:

            values.append(
                confusion[
                    (
                        actual_class,
                        predicted_class
                    )
                ]
            )

        print(
            f"{CLASS_NAMES[actual_class]:<12}"
            f"{values[0]:>10}"
            f"{values[1]:>11}"
            f"{values[2]:>13}"
        )

    # ========================================================
    # MISSED OBJECTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MISSED GROUND-TRUTH OBJECTS")
    print("=" * 70)

    for class_id in [0, 1, 2]:

        print(
            f"{CLASS_NAMES[class_id]:<12}: "
            f"{missed_ground_truth[class_id]}"
        )

    # ========================================================
    # FALSE PREDICTIONS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("UNMATCHED PREDICTIONS / FALSE POSITIVES")
    print("=" * 70)

    for class_id in [0, 1, 2]:

        print(
            f"{CLASS_NAMES[class_id]:<12}: "
            f"{false_predictions[class_id]}"
        )

    # ========================================================
    # TOTALS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TOTALS")
    print("=" * 70)

    print("\nGround-truth objects:")

    for class_id in [0, 1, 2]:

        print(
            f"{CLASS_NAMES[class_id]:<12}: "
            f"{total_ground_truth[class_id]}"
        )

    print("\nPredicted objects:")

    for class_id in [0, 1, 2]:

        print(
            f"{CLASS_NAMES[class_id]:<12}: "
            f"{total_predictions[class_id]}"
        )

    print("\nImages processed:", images_processed)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()