import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import time
import cv2
import torch
from pathlib import Path

from torch_staintools.normalizer import NormalizerBuilder


SOURCE = Path(r"D:\Blood_Smear_AI\data\final_dataset\images\train")
OUTPUT = Path(r"D:\Blood_Smear_AI\data\vahadane_gpu_test")

OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda")

images = sorted([
    p for p in SOURCE.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])[:3]

print("=" * 70)
print("VAHADANE GPU TEST")
print("=" * 70)

print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"CUDA   : {torch.version.cuda}")
print(f"Images : {len(images)}")
print(f"Output : {OUTPUT}")
print()

# ------------------------------------------------------------
# CREATE NORMALIZER
# ------------------------------------------------------------

print("Creating Vahadane normalizer...")

normalizer = NormalizerBuilder.build(
    "vahadane",
    concentration_solver="qr",
    use_cache=False
)

normalizer = normalizer.to(DEVICE)

# ------------------------------------------------------------
# LOAD REFERENCE
# ------------------------------------------------------------

reference = cv2.imread(
    str(images[0]),
    cv2.IMREAD_COLOR
)

reference = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2RGB
)

reference_tensor = (
    torch.from_numpy(reference)
    .permute(2, 0, 1)
    .unsqueeze(0)
    .float()
    / 255.0
)

reference_tensor = reference_tensor.to(DEVICE)

print("Reference device:", reference_tensor.device)
print("Reference shape :", tuple(reference_tensor.shape))

# ------------------------------------------------------------
# FIT
# ------------------------------------------------------------

print()
print("Fitting reference on GPU...")

torch.cuda.synchronize()
start = time.perf_counter()

normalizer.fit(reference_tensor)

torch.cuda.synchronize()
fit_time = time.perf_counter() - start

print(f"Reference fitting time: {fit_time:.2f} seconds")

# ------------------------------------------------------------
# PROCESS 3 IMAGES
# ------------------------------------------------------------

total_start = time.perf_counter()

for i, image_path in enumerate(images, 1):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )

    tensor = tensor.to(DEVICE)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        result = normalizer(tensor)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    result = (
        result
        .squeeze(0)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )

    output = OUTPUT / f"normalized_{i:02d}.jpg"

    cv2.imwrite(
        str(output),
        result,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    print(
        f"{i}/3 | "
        f"{image_path.name} | "
        f"{elapsed:.2f}s"
    )


torch.cuda.synchronize()

total_time = time.perf_counter() - total_start

print()
print("=" * 70)
print("GPU TEST COMPLETE")
print("=" * 70)

print(f"Fit time       : {fit_time:.2f}s")
print(f"3-image time   : {total_time:.2f}s")
print(f"Average/image  : {total_time / 3:.2f}s")

print(
    f"Peak GPU memory: "
    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
)

print()
print(f"Output: {OUTPUT}")