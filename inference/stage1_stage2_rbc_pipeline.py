from pathlib import Path
import argparse
import json

import cv2
from PIL import Image
from ultralytics import YOLO

from stage2_rbc_morphology import RBCMorphologyClassifier

YOLO_MODEL = Path(r"D:\Blood_Smear_AI\runs\FINAL_YOLO11N_3CLASS-3\weights\best.pt")
RBC_MORPHOLOGY_MODEL = Path(r"D:\Blood_Smear_AI\runs\rbc_morphology\weights\best_rbc_morphology.pth")
OUTPUT_DIR = Path(r"D:\Blood_Smear_AI\outputs\rbc_pipeline")
YOLO_CONFIDENCE = 0.25
YOLO_IOU = 0.50
IMAGE_SIZE = 640
RBC_CLASS_ID, WBC_CLASS_ID, PLATELET_CLASS_ID = 0, 1, 2

class BloodSmearPipeline:
    def __init__(self):
        print("=" * 75)
        print("BLOOD SMEAR STAGE 1 + STAGE 2 RBC PIPELINE")
        print("=" * 75)
        if not YOLO_MODEL.exists():
            raise FileNotFoundError(f"YOLO model not found:\n{YOLO_MODEL}")
        if not RBC_MORPHOLOGY_MODEL.exists():
            raise FileNotFoundError(f"RBC morphology model not found:\n{RBC_MORPHOLOGY_MODEL}")
        print("\nLoading Stage 1 YOLO11n...")
        self.detector = YOLO(str(YOLO_MODEL))
        print("Loading Stage 2 RBC morphology classifier...")
        self.morphology = RBCMorphologyClassifier(model_path=RBC_MORPHOLOGY_MODEL)
        print("\nStage 2 morphology classes:")
        for index, name in enumerate(self.morphology.classes):
            print(f"  {index}: {name}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "crops").mkdir(parents=True, exist_ok=True)
        print("\nPipeline loaded successfully.")

    def analyze(self, image_path):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found:\n{image_path}")
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"OpenCV could not read:\n{image_path}")
        annotated = original.copy()
        height, width = original.shape[:2]
        print("\nStage 1: YOLO11n detection...")
        results = self.detector.predict(source=str(image_path), imgsz=IMAGE_SIZE,
                                        conf=YOLO_CONFIDENCE, iou=YOLO_IOU,
                                        device=0, verbose=False)
        result = results[0]
        counts = {"RBC": 0, "WBC": 0, "Platelet": 0}
        detections = []
        rbc_morphology_counts = {}
        if result.boxes is not None:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            for detection_index, (box, detection_confidence, class_id) in enumerate(
                zip(xyxy, confidences, class_ids), start=1
            ):
                x1, y1, x2, y2 = box
                x1 = max(0, min(int(round(x1)), width - 1))
                y1 = max(0, min(int(round(y1)), height - 1))
                x2 = max(x1 + 1, min(int(round(x2)), width))
                y2 = max(y1 + 1, min(int(round(y2)), height))
                if class_id == RBC_CLASS_ID:
                    class_name = "RBC"
                elif class_id == WBC_CLASS_ID:
                    class_name = "WBC"
                elif class_id == PLATELET_CLASS_ID:
                    class_name = "Platelet"
                else:
                    continue
                counts[class_name] += 1
                cell_result = {
                    "id": detection_index,
                    "class": class_name,
                    "detection_confidence": float(detection_confidence),
                    "bbox": [x1, y1, x2, y2]
                }
                if class_id == RBC_CLASS_ID:
                    crop = original[y1:y2, x1:x2]
                    if crop.size == 0:
                        cell_result["morphology_error"] = "Empty RBC crop"
                    else:
                        crop_filename = f"rbc_{detection_index:04d}.jpg"
                        crop_path = OUTPUT_DIR / "crops" / crop_filename
                        cv2.imwrite(str(crop_path), crop)
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_pil = Image.fromarray(crop_rgb)
                        morphology_result = self.morphology.predict(crop_pil)
                        morphology = morphology_result["morphology"]
                        cell_result["morphology"] = morphology
                        cell_result["morphology_confidence"] = float(morphology_result["confidence"])
                        cell_result["morphology_probabilities"] = morphology_result["probabilities"]
                        cell_result["crop"] = str(crop_path)
                        rbc_morphology_counts[morphology] = rbc_morphology_counts.get(morphology, 0) + 1
                if class_name == "RBC":
                    label = f"RBC {detection_confidence:.2f}"
                    if "morphology" in cell_result:
                        label += f" | {cell_result['morphology']} {cell_result['morphology_confidence']:.2f}"
                elif class_name == "WBC":
                    label = f"WBC {detection_confidence:.2f}"
                else:
                    label = f"Platelet {detection_confidence:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                detections.append(cell_result)
        total_cells = sum(counts.values())
        morphology_distribution = {}
        total_rbc = counts["RBC"]
        for morphology, count in rbc_morphology_counts.items():
            morphology_distribution[morphology] = {
                "count": count,
                "percentage": (count / total_rbc * 100) if total_rbc else 0.0
            }
        detection_confidences = [d["detection_confidence"] for d in detections]
        morphology_confidences = [d["morphology_confidence"] for d in detections if "morphology_confidence" in d]
        summary = {
            "input_image": str(image_path),
            "counts": counts,
            "total_cells": total_cells,
            "rbc_morphology": morphology_distribution,
            "detection_confidence": {
                "average": sum(detection_confidences) / len(detection_confidences) if detection_confidences else 0.0,
                "minimum": min(detection_confidences) if detection_confidences else 0.0,
                "maximum": max(detection_confidences) if detection_confidences else 0.0
            },
            "rbc_morphology_confidence": {
                "average": sum(morphology_confidences) / len(morphology_confidences) if morphology_confidences else 0.0,
                "minimum": min(morphology_confidences) if morphology_confidences else 0.0,
                "maximum": max(morphology_confidences) if morphology_confidences else 0.0
            },
            "detections": detections
        }
        stem = image_path.stem
        annotated_path = OUTPUT_DIR / f"{stem}_annotated.jpg"
        json_path = OUTPUT_DIR / f"{stem}_results.json"
        cv2.imwrite(str(annotated_path), annotated)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print("\n" + "=" * 75)
        print("ANALYSIS COMPLETE")
        print("=" * 75)
        print(f"\nTotal cells: {total_cells}")
        print(f"RBC       : {counts['RBC']}")
        print(f"WBC       : {counts['WBC']}")
        print(f"Platelet  : {counts['Platelet']}")
        print("\nRBC morphology:")
        if morphology_distribution:
            for morphology, values in morphology_distribution.items():
                print(f"  {morphology:<20}{values['count']:>5} ({values['percentage']:.2f}%)")
        else:
            print("  No RBC morphology results.")
        print(f"\nAnnotated image:\n{annotated_path}")
        print(f"\nJSON results:\n{json_path}")
        print(f"\nRBC crops:\n{OUTPUT_DIR / 'crops'}")
        return summary

def main():
    parser = argparse.ArgumentParser(description="Stage 1 YOLO11n + Stage 2 RBC morphology pipeline")
    parser.add_argument("--image", required=True, help="Path to a blood-smear image")
    args = parser.parse_args()
    BloodSmearPipeline().analyze(args.image)

if __name__ == "__main__":
    main()
