import os

# Prevent Torch-Inductor compiler issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import cv2
import shutil
import torch
import time

from pathlib import Path
from torch.nn.functional import pad

from torch_staintools.normalizer import NormalizerBuilder


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path(
    r"D:\Blood_Smear_AI\data\final_dataset"
)

OUTPUT = Path(
    r"D:\Blood_Smear_AI\data\final_dataset_vahadane"
)

DEVICE = torch.device("cuda")

BATCH_SIZE = 2

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

SPLITS = [
    "train",
    "val",
    "test"
]


# ============================================================
# GPU CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "Do not run the full normalization."
    )

print("=" * 75)
print("FULL VAHADANE GPU NORMALIZATION")
print("=" * 75)

print("GPU       :", torch.cuda.get_device_name(0))
print("CUDA      :", torch.version.cuda)
print("Torch     :", torch.__version__)
print("Batch     :", BATCH_SIZE)
print("Source    :", SOURCE)
print("Output    :", OUTPUT)
print()


# ============================================================
# CREATE OUTPUT STRUCTURE
# ============================================================

for split in SPLITS:

    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FIND REFERENCE IMAGE
# ============================================================

train_source = SOURCE / "images" / "train"

train_images = sorted([
    p for p in train_source.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
])

if not train_images:
    raise RuntimeError(
        "No training images found."
    )

REFERENCE_IMAGE = train_images[0]

print("Reference image:", REFERENCE_IMAGE.name)
print()


# ============================================================
# CREATE VAHADANE NORMALIZER
# ============================================================

print("Creating Vahadane normalizer...")

normalizer = NormalizerBuilder.build(
    "vahadane",
    concentration_solver="qr",
    use_cache=False
).to(DEVICE)

print("Normalizer moved to GPU.")
print()


# ============================================================
# LOAD REFERENCE
# ============================================================

reference_bgr = cv2.imread(
    str(REFERENCE_IMAGE),
    cv2.IMREAD_COLOR
)

if reference_bgr is None:
    raise RuntimeError(
        f"Could not read reference image: "
        f"{REFERENCE_IMAGE}"
    )

reference_rgb = cv2.cvtColor(
    reference_bgr,
    cv2.COLOR_BGR2RGB
)

reference_tensor = (
    torch.from_numpy(reference_rgb)
    .permute(2, 0, 1)
    .unsqueeze(0)
    .float()
    / 255.0
)

reference_tensor = reference_tensor.to(
    DEVICE
)

print(
    "Reference shape:",
    tuple(reference_tensor.shape)
)

print(
    "Reference device:",
    reference_tensor.device
)

print()
print("Fitting Vahadane reference...")


torch.cuda.synchronize()

fit_start = time.perf_counter()

with torch.no_grad():
    normalizer.fit(reference_tensor)

torch.cuda.synchronize()

fit_time = time.perf_counter() - fit_start

print(
    f"Reference fitting complete: "
    f"{fit_time:.2f} seconds"
)

print()


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    tensor = (
        torch.from_numpy(image_rgb)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    return tensor


# ============================================================
# CREATE PADDED BATCH
# ============================================================

def create_batch(paths):

    tensors = []

    shapes = []

    for path in paths:

        tensor = load_image(path)

        tensors.append(tensor)

        shapes.append(
            (
                tensor.shape[1],
                tensor.shape[2]
            )
        )


    max_height = max(
        x.shape[1]
        for x in tensors
    )

    max_width = max(
        x.shape[2]
        for x in tensors
    )


    padded_images = []

    for tensor in tensors:

        _, height, width = tensor.shape

        pad_height = (
            max_height - height
        )

        pad_width = (
            max_width - width
        )

        padded = pad(
            tensor,
            (
                0,
                pad_width,
                0,
                pad_height
            ),
            mode="replicate"
        )

        padded_images.append(
            padded
        )


    batch = torch.stack(
        padded_images,
        dim=0
    )

    return (
        batch.to(DEVICE),
        shapes
    )


# ============================================================
# SAVE NORMALIZED RESULTS
# ============================================================

def save_results(
    result,
    paths,
    original_shapes,
    output_dir
):

    result = result.clamp(
        0.0,
        1.0
    )

    for i, path in enumerate(paths):

        height, width = original_shapes[i]

        image = result[i]

        # Remove temporary padding
        image = image[
            :,
            :height,
            :width
        ]

        # BCHW -> HWC
        image = (
            image
            .permute(1, 2, 0)
            .mul(255)
            .round()
            .byte()
            .cpu()
            .numpy()
        )

        # RGB -> BGR
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        output_path = (
            output_dir /
            path.name
        )

        if path.suffix.lower() in [
            ".jpg",
            ".jpeg"
        ]:

            success = cv2.imwrite(
                str(output_path),
                image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

        elif path.suffix.lower() == ".png":

            success = cv2.imwrite(
                str(output_path),
                image,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    3
                ]
            )

        else:

            success = cv2.imwrite(
                str(output_path),
                image
            )


        if not success:
            raise IOError(
                f"Could not save: "
                f"{output_path}"
            )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_single(
    image_path,
    output_dir
):

    batch, shapes = create_batch(
        [image_path]
    )

    with torch.no_grad():

        result = normalizer(batch)

    save_results(
        result,
        [image_path],
        shapes,
        output_dir
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

processed_total = 0
skipped_total = 0
failed = []

overall_start = time.perf_counter()


for split in SPLITS:

    print()
    print("=" * 75)
    print(
        f"PROCESSING {split.upper()}"
    )
    print("=" * 75)

    source_dir = (
        SOURCE /
        "images" /
        split
    )

    output_dir = (
        OUTPUT /
        "images" /
        split
    )


    images = sorted([
        p for p in source_dir.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    ])


    total = len(images)

    print(
        f"Total images: {total}"
    )


    # --------------------------------------------------------
    # Remove images already processed
    # --------------------------------------------------------

    pending = []

    for image_path in images:

        output_path = (
            output_dir /
            image_path.name
        )

        if (
            output_path.exists()
            and output_path.stat().st_size > 0
        ):

            skipped_total += 1

        else:

            pending.append(
                image_path
            )


    print(
        f"Already complete: "
        f"{total - len(pending)}"
    )

    print(
        f"Remaining: "
        f"{len(pending)}"
    )

    print()


    # --------------------------------------------------------
    # BATCH PROCESSING
    # --------------------------------------------------------

    for start in range(
        0,
        len(pending),
        BATCH_SIZE
    ):

        batch_paths = pending[
            start:
            start + BATCH_SIZE
        ]


        try:

            batch, shapes = create_batch(
                batch_paths
            )


            torch.cuda.synchronize()

            batch_start = (
                time.perf_counter()
            )


            with torch.no_grad():

                result = normalizer(
                    batch
                )


            torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - batch_start
            )


            save_results(
                result,
                batch_paths,
                shapes,
                output_dir
            )


            processed_total += len(
                batch_paths
            )


            current = (
                start +
                len(batch_paths)
            )


            remaining = (
                len(pending) -
                current
            )


            avg = (
                elapsed /
                len(batch_paths)
            )


            print(
                f"[{split}] "
                f"{current}/{len(pending)} "
                f"| batch={len(batch_paths)} "
                f"| {elapsed:.2f}s "
                f"| {avg:.2f}s/image "
                f"| remaining={remaining}"
            )


            del batch
            del result

            torch.cuda.empty_cache()


        except Exception as batch_error:

            print()
            print(
                "BATCH FAILED"
            )

            print(
                f"Error: {batch_error}"
            )

            print(
                "Retrying images individually..."
            )


            # ------------------------------------------------
            # FALLBACK: ONE IMAGE AT A TIME
            # ------------------------------------------------

            for image_path in batch_paths:

                try:

                    process_single(
                        image_path,
                        output_dir
                    )

                    processed_total += 1

                    print(
                        f"  Recovered: "
                        f"{image_path.name}"
                    )


                except Exception as image_error:

                    failed.append(
                        (
                            split,
                            str(image_path),
                            str(image_error)
                        )
                    )

                    print(
                        f"  FAILED: "
                        f"{image_path.name}"
                    )

                    print(
                        f"  {image_error}"
                    )


# ============================================================
# COPY LABELS
# ============================================================

print()
print("=" * 75)
print("COPYING YOLO LABELS")
print("=" * 75)


for split in SPLITS:

    source_labels = (
        SOURCE /
        "labels" /
        split
    )

    output_labels = (
        OUTPUT /
        "labels" /
        split
    )


    label_files = list(
        source_labels.glob("*.txt")
    )


    copied = 0

    for label_file in label_files:

        destination = (
            output_labels /
            label_file.name
        )

        if not destination.exists():

            shutil.copy2(
                label_file,
                destination
            )

            copied += 1


    print(
        f"{split}: "
        f"{len(label_files)} labels found, "
        f"{copied} copied"
    )


# ============================================================
# COPY DATA YAML
# ============================================================

source_yaml = (
    SOURCE /
    "data.yaml"
)

output_yaml = (
    OUTPUT /
    "data.yaml"
)


if source_yaml.exists():

    shutil.copy2(
        source_yaml,
        output_yaml
    )

    print()
    print(
        "data.yaml copied."
    )

else:

    print()
    print(
        "WARNING: data.yaml missing."
    )


# ============================================================
# FAILURE LOG
# ============================================================

if failed:

    failure_file = (
        OUTPUT /
        "vahadane_failed.txt"
    )

    with open(
        failure_file,
        "w",
        encoding="utf-8"
    ) as f:

        for split, image, error in failed:

            f.write(
                f"{split} | "
                f"{image} | "
                f"{error}\n"
            )

    print()
    print(
        f"FAILED IMAGES: "
        f"{len(failed)}"
    )

    print(
        f"Failure log: "
        f"{failure_file}"
    )

else:

    print()
    print(
        "No normalization failures."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

total_time = (
    time.perf_counter()
    - overall_start
)

print()
print("=" * 75)
print("VAHADANE NORMALIZATION FINISHED")
print("=" * 75)

print(
    f"Newly processed : "
    f"{processed_total}"
)

print(
    f"Already existed : "
    f"{skipped_total}"
)

print(
    f"Failed          : "
    f"{len(failed)}"
)

print(
    f"Total time      : "
    f"{total_time / 3600:.2f} hours"
)

print()
print(
    "Output:"
)

print(
    OUTPUT
)

print("=" * 75)