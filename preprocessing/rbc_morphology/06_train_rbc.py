from pathlib import Path
import json
import random
import csv

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(
    r"D:\Blood_Smear_AI\data\rbc_morphology\prepared"
)

OUTPUT = Path(
    r"D:\Blood_Smear_AI\runs\rbc_morphology"
)

MODEL_OUTPUT = OUTPUT / "weights"

IMAGE_SIZE = 224

BATCH_SIZE = 8

GRADIENT_ACCUMULATION = 2

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 7

NUM_WORKERS = 2

SEED = 42


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


# ============================================================
# TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 75)
    print("RBC MORPHOLOGY - EFFICIENTNET-B0")
    print("=" * 75)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # TRANSFORMS
    # --------------------------------------------------------

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.3
        ),

        transforms.RandomRotation(
            10
        ),

        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.08,
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

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print("\nLoading datasets...")

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
        f"Train: {len(train_dataset)}"
    )

    print(
        f"Val  : {len(val_dataset)}"
    )

    print(
        f"Test : {len(test_dataset)}"
    )

    print(
        f"Classes: {num_classes}"
    )

    print("\nClass mapping:")

    for i, name in enumerate(classes):

        print(
            f"{i}: {name}"
        )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    targets = np.array(
        train_dataset.targets
    )

    class_counts = np.bincount(
        targets,
        minlength=num_classes
    )

    print("\nTraining class counts:")

    for i, name in enumerate(classes):

        print(
            f"{name:<20}"
            f"{class_counts[i]}"
        )

    # --------------------------------------------------------
    # WEIGHTED SAMPLER
    # --------------------------------------------------------

    print("\nCreating weighted sampler...")

    class_weights = (
        len(targets)
        /
        (
            num_classes *
            class_counts
        )
    )

    sample_weights = np.array(
        [
            class_weights[label]
            for label in targets
        ],
        dtype=np.float64
    )

    sample_weights = torch.from_numpy(
        sample_weights
    ).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    print("Creating dataloaders...")

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

    print(
        f"Training batches per epoch: "
        f"{len(train_loader)}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading EfficientNet-B0...")

    weights = EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(
        weights=weights
    )

    in_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    model = model.to(DEVICE)

    print("Model loaded and moved to GPU.")

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    use_amp = (
        DEVICE.type == "cuda"
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp
        else None
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    def evaluate(loader):

        model.eval()

        all_targets = []
        all_predictions = []

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():

            for images, targets_batch in loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                targets_batch = targets_batch.to(
                    DEVICE,
                    non_blocking=True
                )

                if use_amp:

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16
                    ):

                        outputs = model(
                            images
                        )

                        loss = criterion(
                            outputs,
                            targets_batch
                        )

                else:

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        targets_batch
                    )

                predictions = outputs.argmax(
                    dim=1
                )

                batch_size = (
                    targets_batch.size(0)
                )

                total_loss += (
                    loss.item() *
                    batch_size
                )

                total_samples += (
                    batch_size
                )

                all_targets.extend(
                    targets_batch.cpu().numpy()
                )

                all_predictions.extend(
                    predictions.cpu().numpy()
                )

        average_loss = (
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

        return (
            average_loss,
            accuracy,
            precision,
            recall,
            f1,
            all_targets,
            all_predictions
        )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("STARTING TRAINING")
    print("=" * 75)

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0
        samples_seen = 0

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        # ----------------------------------------------------
        # TRAIN BATCHES
        # ----------------------------------------------------

        for batch_index, (
            images,
            targets_batch
        ) in enumerate(
            train_loader
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets_batch = targets_batch.to(
                DEVICE,
                non_blocking=True
            )

            # -----------------------------------------------
            # Forward
            # -----------------------------------------------

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    outputs = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        targets_batch
                    )

                    scaled_loss = (
                        loss /
                        GRADIENT_ACCUMULATION
                    )

            else:

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    targets_batch
                )

                scaled_loss = (
                    loss /
                    GRADIENT_ACCUMULATION
                )

            # -----------------------------------------------
            # Backward
            # -----------------------------------------------

            if use_amp:

                scaler.scale(
                    scaled_loss
                ).backward()

            else:

                scaled_loss.backward()

            # -----------------------------------------------
            # Optimizer
            # -----------------------------------------------

            if (
                (batch_index + 1)
                % GRADIENT_ACCUMULATION
                == 0
            ):

                if use_amp:

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            batch_size = (
                targets_batch.size(0)
            )

            running_loss += (
                loss.item() *
                batch_size
            )

            samples_seen += (
                batch_size
            )

            # -----------------------------------------------
            # Progress
            # -----------------------------------------------

            if (
                (batch_index + 1) % 100 == 0
                or
                (batch_index + 1) == len(train_loader)
            ):

                print(
                    f"  Batch "
                    f"{batch_index + 1}/"
                    f"{len(train_loader)}",
                    end="\r"
                )

        # ----------------------------------------------------
        # Handle leftover gradients
        # ----------------------------------------------------

        if (
            len(train_loader)
            % GRADIENT_ACCUMULATION
            != 0
        ):

            if use_amp:

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            _,
            _
        ) = evaluate(
            val_loader
        )

        scheduler.step()

        train_loss = (
            running_loss /
            samples_seen
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print("\n")

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Acc    : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val Macro P: "
            f"{val_precision:.4f}"
        )

        print(
            f"Val Macro R: "
            f"{val_recall:.4f}"
        )

        print(
            f"Val Macro F1: "
            f"{val_f1:.4f}"
        )

        print(
            f"Learning rate: "
            f"{current_lr:.8f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "learning_rate": current_lr
            }
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "classes":
                        classes,
                    "best_val_f1":
                        best_f1,
                    "model_name":
                        "EfficientNet-B0",
                    "image_size":
                        IMAGE_SIZE
                },
                MODEL_OUTPUT /
                "best_rbc_morphology.pth"
            )

            print(
                ">>> BEST MODEL SAVED"
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        if epoch % 5 == 0:

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "classes":
                        classes
                },
                MODEL_OUTPUT /
                f"epoch_{epoch:02d}.pth"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping after "
                f"{epoch} epochs."
            )

            break

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_file = (
        OUTPUT /
        "training_history.json"
    )

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

    # --------------------------------------------------------
    # LOAD BEST
    # --------------------------------------------------------

    print("\nLoading best model...")

    best_file = (
        MODEL_OUTPUT /
        "best_rbc_morphology.pth"
    )

    checkpoint = torch.load(
        best_file,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("FINAL TEST")
    print("=" * 75)

    (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        test_targets,
        test_predictions
    ) = evaluate(
        test_loader
    )

    print(
        f"\nTest Loss     : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy : {test_accuracy:.4f}"
    )

    print(
        f"Test Macro P  : {test_precision:.4f}"
    )

    print(
        f"Test Macro R  : {test_recall:.4f}"
    )

    print(
        f"Test Macro F1 : {test_f1:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        test_targets,
        test_predictions,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    print("\n")
    print("=" * 75)
    print("PER-CLASS RESULTS")
    print("=" * 75)

    print(report)

    with open(
        OUTPUT /
        "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        test_targets,
        test_predictions
    )

    with open(
        OUTPUT /
        "confusion_matrix.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["class"] + classes
        )

        for name, row in zip(
            classes,
            cm
        ):

            writer.writerow(
                [name] +
                row.tolist()
            )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    summary = {

        "model":
            "EfficientNet-B0",

        "classes":
            classes,

        "train_images":
            len(train_dataset),

        "validation_images":
            len(val_dataset),

        "test_images":
            len(test_dataset),

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_f1,

        "test_loss":
            test_loss,

        "test_accuracy":
            test_accuracy,

        "test_macro_precision":
            test_precision,

        "test_macro_recall":
            test_recall,

        "test_macro_f1":
            test_f1
    }

    with open(
        OUTPUT /
        "final_test_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n")
    print("=" * 75)
    print("TRAINING COMPLETE")
    print("=" * 75)

    print(
        f"\nBest model:\n{best_file}"
    )

    print(
        f"\nResults:\n"
        f"{OUTPUT / 'final_test_results.json'}"
    )


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()