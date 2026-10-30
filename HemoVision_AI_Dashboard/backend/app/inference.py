import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "FINAL_YOLO11N_3CLASS-3"
    / "weights"
    / "best.pt"
)

RBC_MORPH_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "rbc_morphology"
    / "weights"
    / "best_rbc_morphology.pth"
)

WBC_MORPH_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "wbc_morphology"
    / "best_wbc_model.pth"
)

STAGE2_DIR = PROJECT_ROOT / "inference"

if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))


# ---------------------------------------------------------
# CLASSIFIER IMPORTS
# ---------------------------------------------------------

try:
    from stage2_rbc_morphology import RBCMorphologyClassifier
except Exception as exc:
    RBCMorphologyClassifier = None
    RBC_IMPORT_ERROR = str(exc)
else:
    RBC_IMPORT_ERROR = None


try:
    from stage2_wbc_morphology import WBCMorphologyClassifier
except Exception as exc:
    WBCMorphologyClassifier = None
    WBC_IMPORT_ERROR = str(exc)
else:
    WBC_IMPORT_ERROR = None


# ---------------------------------------------------------
# DETECTOR CONFIGURATION
# ---------------------------------------------------------

CLASS_NAMES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet",
}

RBC_CLASS_ID = 0
WBC_CLASS_ID = 1

YOLO_CONFIDENCE = 0.25
YOLO_IOU = 0.50
IMAGE_SIZE = 640

# Ultralytics accepts "cpu" or a CUDA device index.
# Use CPU when CUDA is unavailable.
try:
    import torch

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"


# ---------------------------------------------------------
# LAZY-LOADED MODELS
# ---------------------------------------------------------

_detector = None
_rbc_morphology = None
_wbc_morphology = None


def _load_models():
    global _detector, _rbc_morphology, _wbc_morphology

    if _detector is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO detector weights not found: {MODEL_PATH}"
            )

        _detector = YOLO(str(MODEL_PATH))

    if _rbc_morphology is None:
        if RBCMorphologyClassifier is None:
            raise RuntimeError(
                "Could not import the RBC morphology classifier. "
                f"Import error: {RBC_IMPORT_ERROR}"
            )

        if not RBC_MORPH_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"RBC morphology weights not found: "
                f"{RBC_MORPH_MODEL_PATH}"
            )

        _rbc_morphology = RBCMorphologyClassifier(
            model_path=RBC_MORPH_MODEL_PATH
        )

    if _wbc_morphology is None:
        if WBCMorphologyClassifier is None:
            raise RuntimeError(
                "Could not import the WBC morphology classifier. "
                f"Import error: {WBC_IMPORT_ERROR}"
            )

        if not WBC_MORPH_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"WBC morphology weights not found: "
                f"{WBC_MORPH_MODEL_PATH}"
            )

        _wbc_morphology = WBCMorphologyClassifier(
            model_path=WBC_MORPH_MODEL_PATH
        )


# ---------------------------------------------------------
# IMAGE ENCODING
# ---------------------------------------------------------

def _data_url(image_bgr: np.ndarray) -> str:
    success, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )

    if not success:
        raise RuntimeError("Could not encode annotated image.")

    payload = base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")

    return f"data:image/jpeg;base64,{payload}"


# ---------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------

def run_inference(image_bytes: bytes):
    start = time.perf_counter()

    _load_models()

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("Uploaded image could not be decoded.")

    height, width = image_bgr.shape[:2]
    annotated = image_bgr.copy()

    results = _detector.predict(
        source=image_bgr,
        imgsz=IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        device=DEVICE,
        verbose=False,
    )

    result = results[0]

    counts = {
        "RBC": 0,
        "WBC": 0,
        "Platelet": 0,
    }

    detections = []

    rbc_morphology_counts = {}
    wbc_morphology_counts = {}

    detection_confidences = {
        "RBC": [],
        "WBC": [],
        "Platelet": [],
    }

    rbc_morphology_confidences = []
    wbc_morphology_confidences = []

    # -----------------------------------------------------
    # PROCESS DETECTIONS
    # -----------------------------------------------------

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for idx, (box, conf, class_id) in enumerate(
            zip(boxes, confidences, class_ids),
            start=1,
        ):
            class_id = int(class_id)

            if class_id not in CLASS_NAMES:
                continue

            class_name = CLASS_NAMES[class_id]

            x1, y1, x2, y2 = [
                int(round(value)) for value in box
            ]

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))

            confidence = float(conf)

            counts[class_name] += 1
            detection_confidences[class_name].append(
                confidence
            )

            item = {
                "id": idx,
                "class": class_name,
                "detection_confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            }

            crop = image_bgr[y1:y2, x1:x2]

            # ---------------------------------------------
            # RBC MORPHOLOGY CLASSIFICATION
            # ---------------------------------------------

            if class_id == RBC_CLASS_ID and crop.size:
                crop_rgb = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB,
                )

                crop_pil = Image.fromarray(crop_rgb)

                morph = _rbc_morphology.predict(crop_pil)

                label = morph["morphology"]
                morph_conf = float(morph["confidence"])

                item["morphology"] = label
                item["morphology_confidence"] = morph_conf

                item["morphology_probabilities"] = {
                    str(key): float(value)
                    for key, value
                    in morph["probabilities"].items()
                }

                rbc_morphology_counts[label] = (
                    rbc_morphology_counts.get(label, 0) + 1
                )

                rbc_morphology_confidences.append(
                    morph_conf
                )

            # ---------------------------------------------
            # WBC MORPHOLOGY CLASSIFICATION
            # ---------------------------------------------

            elif class_id == WBC_CLASS_ID and crop.size:
                crop_rgb = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB,
                )

                crop_pil = Image.fromarray(crop_rgb)

                morph = _wbc_morphology.predict(crop_pil)

                label = str(morph["label"])
                morph_conf = float(morph["confidence"])

                item["morphology"] = label
                item["morphology_confidence"] = morph_conf

                item["morphology_probabilities"] = {
                    str(key): float(value)
                    for key, value
                    in morph["probabilities"].items()
                }

                wbc_morphology_counts[label] = (
                    wbc_morphology_counts.get(label, 0) + 1
                )

                wbc_morphology_confidences.append(
                    morph_conf
                )

            # ---------------------------------------------
            # DRAW ANNOTATIONS
            # ---------------------------------------------

            label_text = (
                f"{class_name} {confidence:.2f}"
            )

            if "morphology" in item:
                label_text += (
                    f" | {item['morphology']} "
                    f"{item['morphology_confidence']:.2f}"
                )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                label_text,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            detections.append(item)

    # -----------------------------------------------------
    # AGGREGATE COUNTS
    # -----------------------------------------------------

    total = sum(counts.values())

    percentages = {
        name: round(
            (count / total * 100) if total else 0.0,
            2,
        )
        for name, count in counts.items()
    }

    def build_distribution(class_counts, denominator):
        return {
            name: {
                "count": count,
                "percentage": round(
                    count / denominator * 100
                    if denominator else 0.0,
                    2,
                ),
            }
            for name, count in class_counts.items()
        }

    rbc_distribution = build_distribution(
        rbc_morphology_counts,
        counts["RBC"],
    )

    wbc_distribution = build_distribution(
        wbc_morphology_counts,
        counts["WBC"],
    )

    average_confidence = {
        name: round(
            sum(values) / len(values) if values else 0.0,
            4,
        )
        for name, values in detection_confidences.items()
    }

    avg_rbc_morph = round(
        sum(rbc_morphology_confidences)
        / len(rbc_morphology_confidences)
        if rbc_morphology_confidences else 0.0,
        4,
    )

    avg_wbc_morph = round(
        sum(wbc_morphology_confidences)
        / len(wbc_morphology_confidences)
        if wbc_morphology_confidences else 0.0,
        4,
    )

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {
        "success": True,
        "sample_id": f"PBS-{int(time.time())}",

        "elapsed_seconds": round(
            time.perf_counter() - start,
            3,
        ),

        "image_size": {
            "width": width,
            "height": height,
        },

        "model": {
            "detector": "YOLO11n",
            "detector_weights": str(MODEL_PATH),

            "rbc_morphology": "Existing RBC morphology classifier",
            "rbc_morphology_weights": str(
                RBC_MORPH_MODEL_PATH
            ),

            "wbc_morphology": "EfficientNetV2-S",
            "wbc_morphology_weights": str(
                WBC_MORPH_MODEL_PATH
            ),
        },

        "counts": counts,
        "total_cells": total,
        "percentages": percentages,
        "average_confidence": average_confidence,

        "rbc_morphology_average_confidence": avg_rbc_morph,
        "wbc_morphology_average_confidence": avg_wbc_morph,

        # Preserve the existing RBC morphology response key.
        "morphology": rbc_distribution,

        # Explicitly separate the two morphology tasks.
        "rbc_morphology": rbc_distribution,
        "wbc_morphology": wbc_distribution,

        "detections": detections,

        "annotated_image": _data_url(annotated),

        "inference_policy": {
            "yolo_confidence": YOLO_CONFIDENCE,
            "yolo_iou": YOLO_IOU,
            "device": str(DEVICE),
            "note": (
                "Application thresholds are not official "
                "evaluation metrics. Morphology labels are "
                "model predictions and require validation."
            ),
        },
    }
