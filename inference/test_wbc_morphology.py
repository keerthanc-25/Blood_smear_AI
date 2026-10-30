from pathlib import Path
from PIL import Image

from stage2_wbc_morphology import WBCMorphologyClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    PROJECT_ROOT
    / "data"
    / "wbc_morphology"
    / "prepared"
    / "test"
)

classifier = WBCMorphologyClassifier()

print("Device:", classifier.model.device if hasattr(classifier.model, "device") else "Model loaded")
print("Classes:", classifier.classes)

image_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

test_images = [
    path
    for path in DATASET.rglob("*")
    if path.is_file() and path.suffix.lower() in image_extensions
]

if not test_images:
    raise FileNotFoundError(
        f"No test images found in {DATASET}"
    )

# One image is sufficient for an initial smoke test.
image_path = test_images[0]

with Image.open(image_path) as image:
    result = classifier.predict(image.copy())

print("\nTest image:", image_path)
print("Prediction:", result["label"])
print("Confidence:", round(result["confidence"], 4))
print("Probabilities:")

for class_name, probability in result["probabilities"].items():
    print(f"  {class_name}: {probability:.4f}")

print("\nWBC classifier smoke test completed.")