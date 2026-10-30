import os
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"D:\Blood_Smear_AI"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "runs",
    "blood_smear_rbc_wbc_platelet",
    "weights",
    "best.pt"
)

TEST_IMAGES = os.path.join(
    PROJECT_DIR,
    "data",
    "yolo_dataset_3class",
    "images",
    "test"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "runs",
    "rbc_wbc_platelet_test_predictions"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 60)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nTest images:")
    print(TEST_IMAGES)

    print("\nOutput:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = YOLO(MODEL_PATH)

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    results = model.predict(
        source=TEST_IMAGES,
        imgsz=800,
        conf=0.25,
        iou=0.45,
        device=0,
        save=True,
        save_conf=True,
        project=OUTPUT_DIR,
        name="predictions",
        exist_ok=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("PREDICTIONS COMPLETE")
    print("=" * 60)

    print("\nSaved to:")

    print(
        os.path.join(
            OUTPUT_DIR,
            "predictions"
        )
    )


if __name__ == "__main__":
    main()