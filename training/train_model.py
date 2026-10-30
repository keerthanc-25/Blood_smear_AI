import os
import torch
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = r"D:\Blood_Smear_AI"

# 3-class dataset
DATA_YAML = os.path.join(
    PROJECT_DIR,
    "data",
    "yolo_dataset_3class",
    "data.yaml"
)

# Existing trained model
START_MODEL = os.path.join(
    PROJECT_DIR,
    "runs",
    "blood_smear_rbc_wbc_platelet",
    "weights",
    "best.pt"
)

# Training output
RUNS_DIR = os.path.join(
    PROJECT_DIR,
    "runs"
)

RUN_NAME = "blood_smear_rbc_wbc_platelet_finetuned"


# ============================================================
# GPU / CPU
# ============================================================

if torch.cuda.is_available():

    DEVICE = 0

    print("\nGPU detected:")
    print(torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

else:

    DEVICE = "cpu"

    print("\nWARNING: CUDA is not available.")
    print("Training will use CPU.")


# ============================================================
# INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("YOLO11n RBC / WBC / PLATELET FINE-TUNING")
print("=" * 70)

print("\nDataset:")
print(DATA_YAML)

print("\nStarting model:")
print(START_MODEL)

print("\nOutput:")
print(
    os.path.join(
        RUNS_DIR,
        RUN_NAME
    )
)

print("\nClasses:")
print("0 = RBC")
print("1 = WBC")
print("2 = Platelet")


# ============================================================
# SAFETY CHECKS
# ============================================================

if not os.path.exists(DATA_YAML):

    raise FileNotFoundError(
        f"Dataset YAML not found:\n{DATA_YAML}"
    )


if not os.path.exists(START_MODEL):

    raise FileNotFoundError(
        f"Starting model not found:\n{START_MODEL}"
    )


# ============================================================
# LOAD EXISTING TRAINED MODEL
# ============================================================

print("\nLoading existing trained model...")

model = YOLO(START_MODEL)


# ============================================================
# FINE-TUNING
# ============================================================

print("\nStarting fine-tuning...\n")

results = model.train(

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    data=DATA_YAML,

    # --------------------------------------------------------
    # FINE-TUNING
    # --------------------------------------------------------

    epochs=20,

    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    imgsz=800,

    # --------------------------------------------------------
    # GTX 1050 Ti 4 GB
    # --------------------------------------------------------

    batch=2,

    device=DEVICE,

    workers=0,

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer="AdamW",

    # Lower learning rate because we are fine-tuning
    lr0=0.0003,

    lrf=0.01,

    weight_decay=0.0005,

    cos_lr=True,

    # --------------------------------------------------------
    # LOSS WEIGHTS
    # --------------------------------------------------------

    # Classification is important for
    # RBC vs WBC discrimination
    cls=1.0,

    # Keep bounding-box localization strong
    box=7.5,

    # --------------------------------------------------------
    # AUGMENTATION
    # --------------------------------------------------------

    mosaic=0.8,

    # Turn mosaic off near the end
    close_mosaic=5,

    scale=0.3,

    translate=0.05,

    fliplr=0.5,

    flipud=0.0,

    degrees=0.0,

    shear=0.0,

    perspective=0.0,

    mixup=0.0,

    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    amp=True,

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    patience=6,

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    project=RUNS_DIR,

    name=RUN_NAME,

    exist_ok=True,

    save=True,

    save_period=5,

    plots=True,

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    seed=42,

    verbose=True
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FINE-TUNING COMPLETE")
print("=" * 70)

BEST_MODEL = os.path.join(
    RUNS_DIR,
    RUN_NAME,
    "weights",
    "best.pt"
)

LAST_MODEL = os.path.join(
    RUNS_DIR,
    RUN_NAME,
    "weights",
    "last.pt"
)

print("\nBest model:")
print(BEST_MODEL)

print("\nLast model:")
print(LAST_MODEL)

print("\nNext step:")
print("Evaluate the fine-tuned model on the test set.")