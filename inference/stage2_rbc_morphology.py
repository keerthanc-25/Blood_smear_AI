from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    r"D:\Blood_Smear_AI\runs\rbc_morphology"
    r"\weights\best_rbc_morphology.pth"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([

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
# RBC MORPHOLOGY CLASSIFIER
# ============================================================

class RBCMorphologyClassifier:

    def __init__(
        self,
        model_path=MODEL_PATH,
        device=DEVICE
    ):

        self.device = device

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            model_path,
            map_location=device
        )

        self.classes = checkpoint[
            "classes"
        ]

        num_classes = len(
            self.classes
        )

        # ----------------------------------------------------
        # Build EfficientNet-B0
        # ----------------------------------------------------

        self.model = models.efficientnet_b0(
            weights=None
        )

        in_features = (
            self.model.classifier[1]
            .in_features
        )

        self.model.classifier[1] = (
            nn.Linear(
                in_features,
                num_classes
            )
        )

        # ----------------------------------------------------
        # Load trained weights
        # ----------------------------------------------------

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    # ========================================================
    # PREDICT PIL IMAGE
    # ========================================================

    def predict(
        self,
        image
    ):

        if not isinstance(
            image,
            Image.Image
        ):

            raise TypeError(
                "Expected PIL.Image.Image"
            )

        image = image.convert(
            "RGB"
        )

        tensor = TRANSFORM(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            self.device
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, class_index = (
                probabilities.max(
                    dim=1
                )
            )

        class_index = (
            class_index.item()
        )

        confidence = (
            confidence.item()
        )

        morphology = (
            self.classes[
                class_index
            ]
        )

        # ----------------------------------------------------
        # All class probabilities
        # ----------------------------------------------------

        all_probabilities = {}

        for index, class_name in enumerate(
            self.classes
        ):

            all_probabilities[
                class_name
            ] = float(
                probabilities[
                    0,
                    index
                ].item()
            )

        return {

            "morphology":
                morphology,

            "confidence":
                confidence,

            "class_index":
                class_index,

            "probabilities":
                all_probabilities
        }