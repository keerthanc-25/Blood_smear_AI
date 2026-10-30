from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "wbc_morphology"
    / "best_wbc_model.pth"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class WBCMorphologyClassifier:

    def __init__(self, model_path=MODEL_PATH):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"WBC checkpoint not found: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=False
        )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain model_state_dict."
            )

        self.classes = checkpoint.get("classes")

        if not self.classes:
            raise ValueError(
                "WBC checkpoint does not contain class names."
            )

        self.classes = list(self.classes)

        if len(self.classes) != len(set(self.classes)):
            raise ValueError(
                "Duplicate WBC class names found in checkpoint."
            )

        self.image_size = int(
            checkpoint.get("image_size", IMAGE_SIZE)
        )

        # Match the architecture used during training.
        self.model = models.efficientnet_v2_s(
            weights=None
        )

        in_features = (
            self.model.classifier[-1].in_features
        )

        self.model.classifier[-1] = nn.Linear(
            in_features,
            len(self.classes)
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])

    @torch.inference_mode()
    def predict(self, image):

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(
                "Expected a PIL.Image.Image input."
            )

        tensor = self.transform(image).unsqueeze(0)
        tensor = tensor.to(DEVICE)

        logits = self.model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_index = int(
            probabilities.argmax().item()
        )

        probability_values = probabilities.cpu().tolist()

        return {
            "label": self.classes[predicted_index],
            "confidence": float(
                probability_values[predicted_index]
            ),
            "probabilities": {
                name: float(probability_values[index])
                for index, name in enumerate(self.classes)
            }
        }