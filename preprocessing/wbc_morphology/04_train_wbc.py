from pathlib import Path
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(
    r"D:\Blood_Smear_AI\data\wbc_morphology\prepared"
)

OUTPUT = Path(
    r"D:\Blood_Smear_AI\runs\wbc_morphology"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# TRAINING SETTINGS
# ------------------------------------------------------------

IMAGE_SIZE = 224

# GTX 1050 Ti 4 GB
BATCH_SIZE = 4

EPOCHS = 15

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PATIENCE = 7

SEED = 42

# Print training progress every N batches
PRINT_EVERY = 25

# Resume/checkpoint settings
RESUME = True
LAST_CHECKPOINT = OUTPUT / "last.pt"
BEST_CHECKPOINT = OUTPUT / "best.pt"
LEGACY_BEST_CHECKPOINT = OUTPUT / "best_wbc_model.pth"
HISTORY_FILE = OUTPUT / "training_history.json"
TEST_RESULTS_FILE = OUTPUT / "test_results.json" 


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 75)
print("WBC MORPHOLOGY - MODEL TRAINING")
print("=" * 75)

print(
    f"\nDevice: {DEVICE}",
    flush=True
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}",
        flush=True
    )

    print(
        f"CUDA: {torch.version.cuda}",
        flush=True
    )

    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        flush=True
    )

else:

    print(
        "\nWARNING: CUDA is not available.",
        flush=True
    )

    print(
        "Training will run on CPU.",
        flush=True
    )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASETS
# ============================================================

print("\nLoading datasets...", flush=True)

train_dataset = datasets.ImageFolder(
    DATASET / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATASET / "val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    DATASET / "test",
    transform=eval_transform
)


classes = train_dataset.classes

num_classes = len(classes)


print(
    f"\nClasses: {num_classes}",
    flush=True
)

for index, class_name in enumerate(classes):

    print(
        f"{index:2d} : {class_name}",
        flush=True
    )


# ============================================================
# VERIFY CLASS MAPPING
# ============================================================

if train_dataset.class_to_idx != val_dataset.class_to_idx:

    raise RuntimeError(
        "Train/Val class mapping mismatch."
    )


if train_dataset.class_to_idx != test_dataset.class_to_idx:

    raise RuntimeError(
        "Train/Test class mapping mismatch."
    )


# ============================================================
# CLASS COUNTS
# ============================================================

train_targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    train_targets,
    minlength=num_classes
)


print("\n")
print("=" * 75)
print("TRAINING CLASS COUNTS")
print("=" * 75)

for index, class_name in enumerate(classes):

    print(
        f"{class_name:<25}"
        f"{class_counts[index]:>8}",
        flush=True
    )


# ============================================================
# TEMPERED WEIGHTED SAMPLING
# ============================================================

print("\nCreating weighted sampler...", flush=True)

# Use square-root inverse frequency.
#
# This prevents very small classes from completely
# dominating training.

class_weights = (
    1.0 /
    np.sqrt(
        class_counts.astype(np.float64)
    )
)

sample_weights = class_weights[
    train_targets
]

sample_weights = torch.as_tensor(
    sample_weights,
    dtype=torch.double
)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print("\n")
print("=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print(
    f"Train images : {len(train_dataset)}",
    flush=True
)

print(
    f"Val images   : {len(val_dataset)}",
    flush=True
)

print(
    f"Test images  : {len(test_dataset)}",
    flush=True
)

print(
    f"Batch size   : {BATCH_SIZE}",
    flush=True
)

print(
    f"Train batches: {len(train_loader)}",
    flush=True
)

print(
    f"Epochs       : {EPOCHS}",
    flush=True
)


# ============================================================
# MODEL
# ============================================================

print("\n")
print("=" * 75)
print("LOADING EFFICIENTNETV2-S")
print("=" * 75)

print(
    "\nLoading pretrained EfficientNetV2-S...",
    flush=True
)

weights = (
    models.EfficientNet_V2_S_Weights.DEFAULT
)

model = models.efficientnet_v2_s(
    weights=weights
)


# Replace classifier

in_features = (
    model.classifier[-1].in_features
)

model.classifier[-1] = nn.Linear(
    in_features,
    num_classes
)


model = model.to(DEVICE)


print(
    "Model moved to device successfully.",
    flush=True
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# AMP
# ============================================================

use_amp = DEVICE.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


print(
    f"\nAMP enabled: {use_amp}",
    flush=True
)


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state()
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):
    if not state:
        return

    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    except Exception as exc:
        print(
            f"WARNING: Could not fully restore RNG state: {exc}",
            flush=True
        )


def save_checkpoint(
    path,
    epoch,
    best_macro_f1,
    epochs_without_improvement,
    history
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_macro_f1": best_macro_f1,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "classes": classes,
        "class_to_idx": train_dataset.class_to_idx,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "seed": SEED,
        "rng_state": get_rng_state()
    }

    torch.save(checkpoint, path)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_targets = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=use_amp
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    targets
                )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            batch_size = (
                targets.size(0)
            )

            total_loss += (
                loss.item() *
                batch_size
            )

            total_samples += batch_size

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )


    avg_loss = (
        total_loss /
        total_samples
    )


    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )


    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )


    return {

        "loss": avg_loss,

        "accuracy": accuracy,

        "macro_precision": precision,

        "macro_recall": recall,

        "macro_f1": f1,

        "weighted_precision": weighted_precision,

        "weighted_recall": weighted_recall,

        "weighted_f1": weighted_f1,

        "targets": all_targets,

        "predictions": all_predictions

    }


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 75)
print("STARTING TRAINING")
print("=" * 75)

print(
    "\nTraining will now begin.",
    flush=True
)

print(
    "Progress will be displayed every "
    f"{PRINT_EVERY} batches.",
    flush=True
)


best_macro_f1 = -1.0

epochs_without_improvement = 0

history = []

start_epoch = 1

# ============================================================
# RESUME FROM LAST CHECKPOINT
# ============================================================

if RESUME and LAST_CHECKPOINT.exists():

    print("\n")
    print("=" * 75)
    print("RESUMING FROM LAST CHECKPOINT")
    print("=" * 75)

    print(
        f"Checkpoint: {LAST_CHECKPOINT}",
        flush=True
    )

    checkpoint = torch.load(
        LAST_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if checkpoint.get("classes") != classes:
        raise RuntimeError(
            "Checkpoint class mapping does not match current dataset."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    best_macro_f1 = checkpoint.get(
        "best_macro_f1",
        -1.0
    )

    epochs_without_improvement = checkpoint.get(
        "epochs_without_improvement",
        0
    )

    history = checkpoint.get(
        "history",
        []
    )

    last_completed_epoch = checkpoint.get(
        "epoch",
        0
    )

    start_epoch = last_completed_epoch + 1

    restore_rng_state(
        checkpoint.get("rng_state")
    )

    print(
        f"Last completed epoch: {last_completed_epoch}",
        flush=True
    )

    print(
        f"Next epoch: {start_epoch}/{EPOCHS}",
        flush=True
    )

    print(
        f"Best validation Macro F1: {best_macro_f1:.4f}",
        flush=True
    )

elif RESUME and LEGACY_BEST_CHECKPOINT.exists():

    print("\n")
    print("=" * 75)
    print("LEGACY BEST MODEL FOUND")
    print("=" * 75)

    legacy = torch.load(
        LEGACY_BEST_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if legacy.get("classes") != classes:
        raise RuntimeError(
            "Legacy checkpoint class mapping does not match current dataset."
        )

    model.load_state_dict(
        legacy["model_state_dict"]
    )

    best_macro_f1 = legacy.get(
        "best_macro_f1",
        -1.0
    )

    last_completed_epoch = legacy.get(
        "epoch",
        0
    )

    start_epoch = last_completed_epoch + 1

    print(
        f"Loaded old best model from epoch: {last_completed_epoch}",
        flush=True
    )

    print(
        f"Starting from epoch: {start_epoch}/{EPOCHS}",
        flush=True
    )

    print(
        "NOTE: The old checkpoint did not save optimizer/scheduler state.",
        flush=True
    )

    print(
        "This is a weights-only continuation. Future runs will use true last.pt resume.",
        flush=True
    )

else:

    print(
        "No previous checkpoint found. Starting from epoch 1.",
        flush=True
    )


# ============================================================
# EPOCH LOOP
# ============================================================

for epoch in range(
    start_epoch,
    EPOCHS + 1
):

    epoch_start = time.time()

    model.train()

    running_loss = 0.0

    total_samples = 0


    print("\n")
    print("-" * 75)

    print(
        f"EPOCH {epoch}/{EPOCHS}",
        flush=True
    )

    print("-" * 75)


    # --------------------------------------------------------
    # TRAINING BATCHES
    # --------------------------------------------------------

    for batch_index, (
        images,
        targets
    ) in enumerate(
        train_loader,
        start=1
    ):

        batch_start = time.time()


        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                targets
            )


        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )

        scaler.update()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        batch_size = (
            targets.size(0)
        )

        running_loss += (
            loss.item() *
            batch_size
        )

        total_samples += batch_size


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            batch_index == 1
            or batch_index % PRINT_EVERY == 0
            or batch_index == len(train_loader)
        ):

            elapsed = (
                time.time() -
                batch_start
            )

            current_loss = (
                running_loss /
                total_samples
            )

            progress = (
                batch_index /
                len(train_loader)
            ) * 100


            print(
                f"[Epoch {epoch}/{EPOCHS}] "
                f"Batch {batch_index}/{len(train_loader)} "
                f"({progress:.1f}%) | "
                f"Loss: {current_loss:.4f} | "
                f"Batch time: {elapsed:.2f}s",
                flush=True
            )


    # --------------------------------------------------------
    # TRAIN LOSS
    # --------------------------------------------------------

    train_loss = (
        running_loss /
        total_samples
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\nRunning validation...",
        flush=True
    )

    val_start = time.time()

    val_results = evaluate(
        model,
        val_loader
    )

    val_time = (
        time.time() -
        val_start
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step()


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # EPOCH TIME
    # --------------------------------------------------------

    epoch_time = (
        time.time() -
        epoch_start
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)

    print(
        f"EPOCH {epoch}/{EPOCHS} RESULTS"
    )

    print("=" * 75)

    print(
        f"Train Loss      : "
        f"{train_loss:.4f}"
    )

    print(
        f"Val Loss        : "
        f"{val_results['loss']:.4f}"
    )

    print(
        f"Val Accuracy    : "
        f"{val_results['accuracy']:.4f}"
    )

    print(
        f"Val Macro P     : "
        f"{val_results['macro_precision']:.4f}"
    )

    print(
        f"Val Macro R     : "
        f"{val_results['macro_recall']:.4f}"
    )

    print(
        f"Val Macro F1    : "
        f"{val_results['macro_f1']:.4f}"
    )

    print(
        f"Learning Rate   : "
        f"{current_lr:.7f}"
    )

    print(
        f"Epoch Time      : "
        f"{epoch_time / 60:.2f} min"
    )

    print(
        f"Validation Time : "
        f"{val_time / 60:.2f} min"
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "val_loss":
            val_results["loss"],

        "val_accuracy":
            val_results["accuracy"],

        "val_macro_precision":
            val_results["macro_precision"],

        "val_macro_recall":
            val_results["macro_recall"],

        "val_macro_f1":
            val_results["macro_f1"],

        "learning_rate":
            current_lr,

        "epoch_time_seconds":
            epoch_time

    })


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if (
        val_results["macro_f1"]
        >
        best_macro_f1
    ):

        best_macro_f1 = (
            val_results["macro_f1"]
        )

        epochs_without_improvement = 0


        save_checkpoint(
            BEST_CHECKPOINT,
            epoch,
            best_macro_f1,
            epochs_without_improvement,
            history
        )

        # Keep the old filename for compatibility.
        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    classes,

                "class_to_idx":
                    train_dataset.class_to_idx,

                "best_macro_f1":
                    best_macro_f1,

                "epoch":
                    epoch,

                "image_size":
                    IMAGE_SIZE
            },
            LEGACY_BEST_CHECKPOINT
        )


        print("\n")
        print(
            "*** BEST MODEL SAVED ***",
            flush=True
        )

        print(
            f"Best Macro F1: "
            f"{best_macro_f1:.4f}",
            flush=True
        )


    else:

        epochs_without_improvement += 1


        print(
            f"\nNo improvement for "
            f"{epochs_without_improvement} "
            f"epoch(s).",
            flush=True
        )


    # --------------------------------------------------------
    # SAVE LAST CHECKPOINT
    # --------------------------------------------------------
    # This checkpoint is used for true resume.
    # It contains model, optimizer, scheduler, AMP scaler,
    # history, epoch number, and RNG state.
    # --------------------------------------------------------

    save_checkpoint(
        LAST_CHECKPOINT,
        epoch,
        best_macro_f1,
        epochs_without_improvement,
        history
    )

    print(
        f"LAST CHECKPOINT SAVED: {LAST_CHECKPOINT}",
        flush=True
    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered.",
            flush=True
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = HISTORY_FILE


with open(
    history_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


print(
    f"\nTraining history saved to:",
    flush=True
)

print(
    history_file,
    flush=True
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n")
print("=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)


best_model_path = (
    BEST_CHECKPOINT
    if BEST_CHECKPOINT.exists()
    else LEGACY_BEST_CHECKPOINT
)


if not best_model_path.exists():

    raise FileNotFoundError(
        "Best model was not created."
    )


checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


print(
    "\nBest model loaded.",
    flush=True
)

print(
    f"Best validation Macro F1: "
    f"{checkpoint['best_macro_f1']:.4f}",
    flush=True
)


print(
    "\nRunning test evaluation...",
    flush=True
)


test_results = evaluate(
    model,
    test_loader
)


# ============================================================
# TEST RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FINAL WBC MORPHOLOGY TEST RESULTS")
print("=" * 75)


print(
    f"\nTest Accuracy       : "
    f"{test_results['accuracy']:.4f}"
)

print(
    f"Test Macro Precision: "
    f"{test_results['macro_precision']:.4f}"
)

print(
    f"Test Macro Recall   : "
    f"{test_results['macro_recall']:.4f}"
)

print(
    f"Test Macro F1       : "
    f"{test_results['macro_f1']:.4f}"
)

print(
    f"Test Weighted F1    : "
    f"{test_results['weighted_f1']:.4f}"
)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

test_output = {

    "accuracy":
        test_results["accuracy"],

    "macro_precision":
        test_results["macro_precision"],

    "macro_recall":
        test_results["macro_recall"],

    "macro_f1":
        test_results["macro_f1"],

    "weighted_precision":
        test_results["weighted_precision"],

    "weighted_recall":
        test_results["weighted_recall"],

    "weighted_f1":
        test_results["weighted_f1"],

    "classes":
        classes,

    "best_epoch":
        checkpoint["epoch"],

    "best_validation_macro_f1":
        checkpoint["best_macro_f1"]

}


test_results_file = TEST_RESULTS_FILE


with open(
    test_results_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        test_output,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 75)
print("WBC MORPHOLOGY TRAINING COMPLETE")
print("=" * 75)


print(
    f"\nBest model:"
    f"\n{best_model_path}"
)

print(
    f"\nResume checkpoint:"
    f"\n{LAST_CHECKPOINT}"
)

print(
    f"\nTraining history:"
    f"\n{history_file}"
)

print(
    f"\nTest results:"
    f"\n{test_results_file}"
)

print(
    "\nAll processing completed successfully.",
    flush=True
)