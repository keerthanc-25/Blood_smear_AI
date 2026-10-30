from ultralytics import YOLO
import torch

# ============================================================
# CONFIG
# ============================================================

DATA = r"D:\Blood_Smear_AI\data\final_dataset\data.yaml"
MODEL = "yolo11n.pt"

PROJECT = r"D:\Blood_Smear_AI\runs"
NAME = "FINAL_YOLO11N_3CLASS"

DEVICE = 0

print("=" * 70)
print("FINAL BLOOD SMEAR YOLO11n TRAINING")
print("=" * 70)

print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))

print()
print("Dataset :", DATA)
print("Classes : RBC / WBC / Platelet")
print("=" * 70)


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

model = YOLO(MODEL)


# ============================================================
# FINAL TRAINING
# ============================================================

results = model.train(

    # Dataset
    data=DATA,

    # Model / hardware
    imgsz=640,
    batch=4,
    device=DEVICE,
    workers=0,

    # Training duration
    epochs=30,
    patience=12,

    # Optimizer
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01,
    weight_decay=0.0005,

    # Learning-rate schedule
    cos_lr=True,

    # Loss balance
    box=7.5,
    cls=1.0,
    dfl=1.5,

    # Augmentation
    mosaic=0.7,
    close_mosaic=10,

    scale=0.3,
    translate=0.05,

    fliplr=0.5,
    flipud=0.0,

    degrees=0.0,
    shear=0.0,
    perspective=0.0,

    mixup=0.0,

    # Performance
    amp=True,
    cache=False,

    # Reproducibility
    seed=42,

    # Saving
    project=PROJECT,
    name=NAME,

    save=True,
    save_period=1,

    plots=True,
    verbose=True
)


print()
print("=" * 70)
print("FINAL TRAINING COMPLETE")
print("=" * 70)

print(
    r"Best model:"
    r" D:\Blood_Smear_AI\runs\FINAL_YOLO11N_3CLASS\weights\best.pt"
)

print("=" * 70)