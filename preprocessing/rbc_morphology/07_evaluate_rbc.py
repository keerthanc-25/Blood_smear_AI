from pathlib import Path
import json
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"D:\Blood_Smear_AI")

DATASET = ROOT / "data" / "rbc_morphology" / "prepared"

RUN_DIR = ROOT / "runs" / "rbc_morphology"

OUTPUT = RUN_DIR / "evaluation"

MODEL_PATH = RUN_DIR / "weights" / "best_rbc_morphology.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0

OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# VALIDATE PATHS
# ============================================================

print("=" * 70)
print("RBC MORPHOLOGY - HELD-OUT TEST EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"RBC morphology model not found:\n{MODEL_PATH}"
    )

if not (DATASET / "test").is_dir():
    raise FileNotFoundError(
        f"Test dataset not found:\n{DATASET / 'test'}"
    )

print(f"Model: {MODEL_PATH}")
print(f"Dataset: {DATASET}")

# ============================================================
# LOAD TEST DATA
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_dataset = datasets.ImageFolder(
    DATASET / "test",
    transform=transform,
)

if len(test_dataset) == 0:
    raise RuntimeError("The RBC test dataset is empty.")

classes = test_dataset.classes
num_classes = len(classes)

print(f"\nTest images: {len(test_dataset)}")
print(f"Number of classes: {num_classes}")

for index, class_name in enumerate(classes):
    print(f"{index}: {class_name}")

# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

if not isinstance(checkpoint, dict):
    raise RuntimeError(
        "Unexpected checkpoint format. "
        "Expected a dictionary containing model_state_dict."
    )

if "model_state_dict" not in checkpoint:
    raise RuntimeError(
        "The checkpoint does not contain model_state_dict. "
        "Inspect the checkpoint format before proceeding."
    )

saved_classes = checkpoint.get("classes")

if saved_classes is not None and list(saved_classes) != classes:
    raise RuntimeError(
        "Model classes do not match the test dataset.\n"
        f"Model classes: {saved_classes}\n"
        f"Dataset classes: {classes}"
    )

# ============================================================
# BUILD MODEL
# ============================================================
# This architecture must match the RBC training architecture.
# The checkpoint filename alone does not establish its architecture.
# ============================================================

print("\nBuilding EfficientNet-B0...")

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    num_classes,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("RBC morphology model loaded successfully.")

# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# ============================================================
# PREDICTIONS
# ============================================================

all_targets = []
all_predictions = []
all_probabilities = []

print("\nEvaluating held-out test images...")

with torch.inference_mode():
    for batch_index, (images, targets) in enumerate(
        loader,
        start=1,
    ):
        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = outputs.argmax(dim=1)

        all_targets.extend(
            targets.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        if (
            batch_index % 100 == 0
            or batch_index == len(loader)
        ):
            print(
                f"Processed {batch_index}/{len(loader)} batches",
                flush=True,
            )

y_true = np.asarray(all_targets)
y_pred = np.asarray(all_predictions)
y_prob = np.asarray(all_probabilities)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )
)

macro_p, macro_r, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
)

weighted_p, weighted_r, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
)

row_totals = cm.sum(axis=1, keepdims=True)

normalized_cm = np.divide(
    cm,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0,
)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=classes,
    digits=4,
    zero_division=0,
)

# ============================================================
# SAVE CLASSIFICATION REPORTS
# ============================================================

(OUTPUT / "classification_report.txt").write_text(
    report_text,
    encoding="utf-8",
)

with open(
    OUTPUT / "classification_report.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        report_dict,
        file,
        indent=4,
    )

with open(
    OUTPUT / "classification_report.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "class",
        "precision",
        "recall",
        "f1_score",
        "support",
    ])

    for i, class_name in enumerate(classes):
        writer.writerow([
            class_name,
            float(precision[i]),
            float(recall[i]),
            float(f1[i]),
            int(support[i]),
        ])

# ============================================================
# CONFUSION MATRICES
# ============================================================

def plot_confusion_matrix(
    matrix,
    title,
    filename,
    normalized=False,
):
    fig_width = max(
        10,
        num_classes * 0.85,
    )

    fig_height = max(
        8,
        num_classes * 0.75,
    )

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1 if normalized else None,
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted Class",
        ylabel="True Class",
        title=title,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        0.5
        if normalized
        else matrix.max() / 2 if matrix.size else 0
    )

    for i in range(num_classes):
        for j in range(num_classes):
            value = matrix[i, j]

            text = (
                f"{value:.2f}"
                if normalized
                else f"{int(value)}"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if value < threshold
                    else "black"
                ),
                fontsize=8,
            )

    fig.tight_layout()

    fig.savefig(
        OUTPUT / filename,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


plot_confusion_matrix(
    cm,
    "RBC Morphology - Confusion Matrix",
    "confusion_matrix.png",
)

plot_confusion_matrix(
    normalized_cm,
    "RBC Morphology - Normalized Confusion Matrix",
    "confusion_matrix_normalized.png",
    normalized=True,
)

# ============================================================
# PER-CLASS METRICS GRAPH
# ============================================================

x = np.arange(num_classes)
width = 0.25

fig, ax = plt.subplots(
    figsize=(max(12, num_classes * 1.1), 7)
)

ax.bar(
    x - width,
    precision,
    width,
    label="Precision",
)

ax.bar(
    x,
    recall,
    width,
    label="Recall",
)

ax.bar(
    x + width,
    f1,
    width,
    label="F1-score",
)

ax.set_ylabel("Score")
ax.set_xlabel("RBC Morphology Class")

ax.set_title(
    "RBC Morphology - Per-Class Evaluation Metrics"
)

ax.set_xticks(x)
ax.set_xticklabels(
    classes,
    rotation=45,
    ha="right",
)

ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUTPUT / "per_class_metrics.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# OVERALL METRICS GRAPH
# ============================================================

metric_names = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted Precision",
    "Weighted Recall",
    "Weighted F1",
]

metric_values = [
    accuracy,
    macro_p,
    macro_r,
    macro_f1,
    weighted_p,
    weighted_r,
    weighted_f1,
]

fig, ax = plt.subplots(
    figsize=(12, 6)
)

bars = ax.bar(
    metric_names,
    metric_values,
)

ax.set_title(
    "RBC Morphology - Overall Test Metrics"
)

ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)

ax.tick_params(
    axis="x",
    rotation=30,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

for bar, value in zip(
    bars,
    metric_values,
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()

fig.savefig(
    OUTPUT / "overall_metrics.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# SAVE NUMERIC RESULTS
# ============================================================

summary = {
    "model_path": str(MODEL_PATH),
    "test_dataset": str(DATASET / "test"),
    "test_image_count": len(test_dataset),
    "classes": classes,
    "best_epoch": checkpoint.get("epoch"),
    "best_validation_macro_f1": checkpoint.get(
        "best_macro_f1"
    ),
    "accuracy": float(accuracy),
    "macro_precision": float(macro_p),
    "macro_recall": float(macro_r),
    "macro_f1": float(macro_f1),
    "weighted_precision": float(weighted_p),
    "weighted_recall": float(weighted_r),
    "weighted_f1": float(weighted_f1),
    "per_class": {
        class_name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }
        for i, class_name in enumerate(classes)
    },
}

with open(
    OUTPUT / "overall_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=4,
    )

(OUTPUT / "evaluation_summary.txt").write_text(
    "RBC MORPHOLOGY - HELD-OUT TEST RESULTS\n"
    "======================================\n"
    f"Test images: {len(test_dataset)}\n"
    f"Accuracy: {accuracy:.4f}\n"
    f"Macro precision: {macro_p:.4f}\n"
    f"Macro recall: {macro_r:.4f}\n"
    f"Macro F1: {macro_f1:.4f}\n"
    f"Weighted precision: {weighted_p:.4f}\n"
    f"Weighted recall: {weighted_r:.4f}\n"
    f"Weighted F1: {weighted_f1:.4f}\n\n"
    "PER-CLASS REPORT\n"
    "----------------\n"
    f"{report_text}\n",
    encoding="utf-8",
)

np.savez_compressed(
    OUTPUT / "test_predictions.npz",
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
    classes=np.asarray(classes),
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RBC MORPHOLOGY - TEST RESULTS")
print("=" * 70)

print(f"Test images        : {len(test_dataset)}")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Macro precision    : {macro_p:.4f}")
print(f"Macro recall       : {macro_r:.4f}")
print(f"Macro F1           : {macro_f1:.4f}")
print(f"Weighted precision : {weighted_p:.4f}")
print(f"Weighted recall    : {weighted_r:.4f}")
print(f"Weighted F1        : {weighted_f1:.4f}")

print("\nPer-class report:\n")
print(report_text)

print("\nEvaluation files saved to:")
print(OUTPUT)

print("\nEvaluation completed successfully.")