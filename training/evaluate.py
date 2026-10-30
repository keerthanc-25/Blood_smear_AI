import os
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"D:\Blood_Smear_AI"

# ------------------------------------------------------------
# FINE-TUNED MODEL
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "runs",
    "blood_smear_rbc_wbc_platelet_finetuned",
    "weights",
    "best.pt"
)

# ------------------------------------------------------------
# SAME 3-CLASS DATASET
# ------------------------------------------------------------

DATA_YAML = os.path.join(
    PROJECT_DIR,
    "data",
    "yolo_dataset_3class",
    "data.yaml"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TESTING FINE-TUNED RBC / WBC / PLATELET MODEL")
    print("=" * 70)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nDataset:")
    print(DATA_YAML)

    print("\nTest split:")
    print("data/yolo_dataset_3class/images/test")

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nFine-tuned model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(DATA_YAML):

        raise FileNotFoundError(
            f"\nDataset YAML not found:\n{DATA_YAML}"
        )

    # --------------------------------------------------------
    # Load fine-tuned model
    # --------------------------------------------------------

    print("\nLoading fine-tuned model...")

    model = YOLO(MODEL_PATH)

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------

    print("\nRunning evaluation on TEST set...")
    print("The test set is NOT modified.")

    results = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=800,
        batch=2,
        workers=0,
        device=0,
        plots=True,
        save_json=True,
        verbose=True
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS - FINE-TUNED MODEL")
    print("=" * 70)

    print(f"\nPrecision : {results.box.mp:.4f}")
    print(f"Recall    : {results.box.mr:.4f}")
    print(f"mAP50     : {results.box.map50:.4f}")
    print(f"mAP50-95  : {results.box.map:.4f}")

    # --------------------------------------------------------
    # Overall F1
    # --------------------------------------------------------

    precision = results.box.mp
    recall = results.box.mr

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    print(f"F1        : {f1:.4f}")

    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("PER-CLASS TEST RESULTS")
    print("=" * 70)

    class_names = {
        0: "RBC",
        1: "WBC",
        2: "Platelet"
    }

    for class_id, name in class_names.items():

        if class_id < len(results.box.p):

            precision = results.box.p[class_id]
            recall = results.box.r[class_id]
            map50 = results.box.ap50[class_id]
            map5095 = results.box.ap[class_id]

            # Class F1
            if precision + recall > 0:

                class_f1 = (
                    2 * precision * recall
                    / (precision + recall)
                )

            else:

                class_f1 = 0.0

            print(f"\n{name}")

            print(
                f"  Precision : {precision:.4f}"
            )

            print(
                f"  Recall    : {recall:.4f}"
            )

            print(
                f"  mAP50     : {map50:.4f}"
            )

            print(
                f"  mAP50-95  : {map5095:.4f}"
            )

            print(
                f"  F1        : {class_f1:.4f}"
            )

    # ========================================================
    # MODEL LOCATION
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    print("\nEvaluated model:")
    print(MODEL_PATH)

    print("\nDataset:")
    print(DATA_YAML)

    print("\nClasses:")
    print("0 = RBC")
    print("1 = WBC")
    print("2 = Platelet")

    print("\nTest set:")
    print("154 images")

    print("\n" + "=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()