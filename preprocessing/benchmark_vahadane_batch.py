import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import time
import cv2
import torch
from pathlib import Path

from torch_staintools.normalizer import NormalizerBuilder


SOURCE = Path(
    r"D:\Blood_Smear_AI\data\final_dataset\images\train"
)

DEVICE = torch.device("cuda")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

images = sorted([
    p for p in SOURCE.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
])[:12]


print("=" * 70)
print("VAHADANE BATCH SPEED TEST")
print("=" * 70)

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
print("Images:", len(images))
print()


# ============================================================
# NORMALIZER
# ============================================================

print("Creating normalizer...")

normalizer = NormalizerBuilder.build(
    "vahadane",
    concentration_solver="qr",
    use_cache=False
).to(DEVICE)


# ============================================================
# REFERENCE
# ============================================================

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
).to(DEVICE)


print("Fitting reference...")

torch.cuda.synchronize()
start = time.perf_counter()

normalizer.fit(reference_tensor)

torch.cuda.synchronize()

print(
    f"Reference fit: "
    f"{time.perf_counter() - start:.2f}s"
)

print()


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Could not read {path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    return tensor


# ============================================================
# CREATE PADDED BATCH
# ============================================================

def make_batch(paths):

    tensors = [
        load_image(p)
        for p in paths
    ]

    max_h = max(
        x.shape[1]
        for x in tensors
    )

    max_w = max(
        x.shape[2]
        for x in tensors
    )

    batch = []

    shapes = []

    for tensor in tensors:

        c, h, w = tensor.shape

        shapes.append((h, w))

        pad_h = max_h - h
        pad_w = max_w - w

        # Replicate edge pixels.
        # This avoids artificial black borders.
        padded = torch.nn.functional.pad(
            tensor,
            (0, pad_w, 0, pad_h),
            mode="replicate"
        )

        batch.append(padded)

    batch = torch.stack(batch)

    return batch.to(DEVICE), shapes


# ============================================================
# BENCHMARK
# ============================================================

for batch_size in [2, 4]:

    print("=" * 70)
    print(f"TESTING BATCH SIZE: {batch_size}")
    print("=" * 70)

    times = []

    for start_index in range(
        0,
        len(images),
        batch_size
    ):

        batch_paths = images[
            start_index:
            start_index + batch_size
        ]

        batch, shapes = make_batch(
            batch_paths
        )

        print(
            f"Batch "
            f"{start_index // batch_size + 1}: "
            f"{len(batch_paths)} images | "
            f"shape={tuple(batch.shape)}"
        )

        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():

            result = normalizer(batch)

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        times.append(elapsed)

        print(
            f"  Time: {elapsed:.2f}s"
        )

        print(
            f"  Per image: "
            f"{elapsed / len(batch_paths):.2f}s"
        )

        del batch
        del result

        torch.cuda.empty_cache()


    total_images = len(images)

    total_time = sum(times)

    print()
    print(
        f"Batch {batch_size} total: "
        f"{total_time:.2f}s"
    )

    print(
        f"Batch {batch_size} average/image: "
        f"{total_time / total_images:.2f}s"
    )

    print(
        f"Peak VRAM: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )

    torch.cuda.reset_peak_memory_stats()

    print()


print("=" * 70)
print("BATCH BENCHMARK COMPLETE")
print("=" * 70)