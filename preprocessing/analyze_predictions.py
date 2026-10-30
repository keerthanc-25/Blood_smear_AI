from pathlib import Path
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

OUTPUT_CSV = Path(
    r"D:\Blood_Smear_AI\data\raw_prediction_analysis.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff", ".webp"
}

CLASS_NAMES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RAW PREDICTION CONFIDENCE ANALYSIS")
print("=" * 70)

print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded.")


# ============================================================
# FIND IMAGES
# ============================================================

images = sorted([
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
])

print(f"\nImages found: {len(images)}")


# ============================================================
# STORAGE
# ============================================================

all_predictions = []

class_counts = {
    0: 0,
    1: 0,
    2: 0
}

confidence_values = {
    0: [],
    1: [],
    2: []
}


# ============================================================
# PREDICT
# ============================================================

for index, image_path in enumerate(images, start=1):

    print(
        f"\rProcessing [{index}/{len(images)}] "
        f"{image_path.name}",
        end=""
    )

    results = model.predict(
        source=str(image_path),
        imgsz=800,
        conf=0.05,
        iou=0.45,
        device=0,
        verbose=False
    )

    result = results[0]

    if result.boxes is None:
        continue

    for box in result.boxes:

        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        if class_id not in CLASS_NAMES:
            continue

        class_counts[class_id] += 1
        confidence_values[class_id].append(confidence)

        all_predictions.append([
            image_path.name,
            CLASS_NAMES[class_id],
            class_id,
            confidence
        ])


print("\n")


# ============================================================
# SAVE RAW CSV
# ============================================================

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image",
        "class",
        "class_id",
        "confidence"
    ])

    writer.writerows(all_predictions)


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

print("=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

for class_id, class_name in CLASS_NAMES.items():

    values = confidence_values[class_id]

    print(f"\n{class_name}")
    print("-" * 40)

    print(f"Total predictions: {len(values)}")

    if not values:
        print("No predictions found.")
        continue

    # --------------------------------------------------------
    # Threshold counts
    # --------------------------------------------------------

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95
    ]

    for threshold in thresholds:

        count = sum(
            confidence >= threshold
            for confidence in values
        )

        print(
            f"Confidence >= {threshold:.2f}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print(
        f"Minimum confidence: {min(values):.4f}"
    )

    print(
        f"Maximum confidence: {max(values):.4f}"
    )

    print(
        f"Average confidence: {sum(values)/len(values):.4f}"
    )


# ============================================================
# IMPORTANT RBC ANALYSIS
# ============================================================

rbc_values = confidence_values[0]

print("\n")
print("=" * 70)
print("RBC-SPECIFIC ANALYSIS")
print("=" * 70)

if not rbc_values:

    print("\nWARNING: The model produced ZERO RBC predictions")
    print("even with confidence threshold = 0.05.")

    print(
        "\nThis means lowering the pseudo-label threshold "
        "would NOT solve the RBC problem."
    )

else:

    print(
        f"\nTotal raw RBC predictions: {len(rbc_values)}"
    )

    print("\nRBC confidence distribution:")

    ranges = [
        (0.05, 0.20),
        (0.20, 0.30),
        (0.30, 0.40),
        (0.40, 0.50),
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.85),
        (0.85, 0.90),
        (0.90, 0.95),
        (0.95, 1.01),
    ]

    for low, high in ranges:

        count = sum(
            low <= confidence < high
            for confidence in rbc_values
        )

        print(
            f"{low:.2f} - {high:.2f}: {count}"
        )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nDetailed CSV:")
print(OUTPUT_CSV)

print("\nDo NOT change the confidence threshold yet.")
print("Use the RBC distribution above to decide the next step.")
print("=" * 70)