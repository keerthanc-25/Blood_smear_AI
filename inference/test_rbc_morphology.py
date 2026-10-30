from pathlib import Path

from PIL import Image

from stage2_rbc_morphology import (
    RBCMorphologyClassifier
)


# ============================================================
# TEST IMAGE
# ============================================================

TEST_IMAGE = Path(
    r"D:\Blood_Smear_AI\data\rbc_morphology"
    r"\prepared\test\normal"
)

# Automatically select first image
images = list(
    TEST_IMAGE.glob("*")
)

if not images:

    raise FileNotFoundError(
        f"No images found in:\n{TEST_IMAGE}"
    )

image_path = images[0]


# ============================================================
# LOAD MODEL
# ============================================================

classifier = (
    RBCMorphologyClassifier()
)


# ============================================================
# PREDICT
# ============================================================

image = Image.open(
    image_path
)

result = classifier.predict(
    image
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 65)
print("RBC MORPHOLOGY INFERENCE TEST")
print("=" * 65)

print(
    f"\nImage:\n{image_path}"
)

print(
    f"\nPrediction:"
)

print(
    f"  Morphology : "
    f"{result['morphology']}"
)

print(
    f"  Confidence : "
    f"{result['confidence']:.4f}"
)

print(
    "\nClass probabilities:"
)

for name, probability in sorted(
    result["probabilities"].items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"  {name:<20}"
        f"{probability:.4f}"
    )