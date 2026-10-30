import os
import csv
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\Blood_Smear_AI"

SOURCE_ROOT = r"D:\blood smear images"

LABEL_ROOT = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "labels"
)

OUTPUT_MANIFEST = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "annotation_manifest.csv"
)

# Number of parallel workers.
# 8 is normally a good choice for Windows.
WORKERS = 8


# ============================================================
# FINAL CLASSES
# ============================================================

CLASS_MAP = {
    "RBC": 0,
    "WBC": 1,
    "Platelet": 2,
    "Malaria_Parasite": 3
}


# ============================================================
# CLASS NORMALIZATION
# ============================================================

def normalize_class(name):

    name = str(name).strip().lower()

    if name in {
        "rbc",
        "red blood cell",
        "red_blood_cell",
        "erythrocyte",
        "erythroblast"
    }:
        return "RBC"

    if name in {
        "wbc",
        "white blood cell",
        "white_blood_cell",
        "lymphocyte",
        "lymphocyte_large_granular",
        "lymphocyte_neoplastic",
        "lymphocyte_reactive",
        "monocyte",
        "neutrophil",
        "segmented neutrophil",
        "eosinophil",
        "basophil",
        "metamyelocyte",
        "myelocyte",
        "promyelocyte",
        "blast",
        "plasma cell",
        "plasma_cell",
        "hairy cell",
        "hairy_cell",
        "smudge cell",
        "smudge_cell",
        "normoblast"
    }:
        return "WBC"

    if name in {
        "platelet",
        "giant platelet",
        "giant_platelet"
    }:
        return "Platelet"

    if name in {
        "parasitized",
        "parasite",
        "malaria parasite",
        "malaria_parasite",
        "plasmodium"
    }:
        return "Malaria_Parasite"

    return None


# ============================================================
# YOLO BOX
# ============================================================

def yolo_box(
    xmin,
    ymin,
    xmax,
    ymax,
    width,
    height
):

    if width <= 0 or height <= 0:
        return None

    xmin = max(0.0, min(float(xmin), width))
    xmax = max(0.0, min(float(xmax), width))

    ymin = max(0.0, min(float(ymin), height))
    ymax = max(0.0, min(float(ymax), height))

    if xmax <= xmin or ymax <= ymin:
        return None

    xc = ((xmin + xmax) / 2) / width
    yc = ((ymin + ymax) / 2) / height

    w = (xmax - xmin) / width
    h = (ymax - ymin) / height

    return xc, yc, w, h


# ============================================================
# NIH TXT
# ============================================================

def parse_nih_txt(annotation_file):

    annotations = []

    try:

        with open(
            annotation_file,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if not lines:
            return []

        header = lines[0].split(",")

        if len(header) < 3:
            return []

        try:
            width = int(header[1])
            height = int(header[2])
        except ValueError:
            return []

        for line in lines[1:]:

            parts = line.split(",")

            if len(parts) < 9:
                continue

            biological_class = normalize_class(
                parts[1]
            )

            if biological_class is None:
                continue

            try:

                x1 = float(parts[5])
                y1 = float(parts[6])
                x2 = float(parts[7])
                y2 = float(parts[8])

            except ValueError:
                continue

            # NIH malaria annotations are circles.
            radius = abs(x2 - x1) / 2

            center_x = (x1 + x2) / 2
            center_y = y1

            xmin = center_x - radius
            xmax = center_x + radius

            ymin = center_y - radius
            ymax = center_y + radius

            box = yolo_box(
                xmin,
                ymin,
                xmax,
                ymax,
                width,
                height
            )

            if box is None:
                continue

            annotations.append(
                (
                    CLASS_MAP[biological_class],
                    *box
                )
            )

        return annotations

    except Exception:
        return []


# ============================================================
# VOC XML
# ============================================================

def parse_voc_xml(annotation_file):

    annotations = []

    try:

        tree = ET.parse(annotation_file)
        root = tree.getroot()

        size = root.find("size")

        if size is None:
            return []

        width = float(
            size.findtext("width", "0")
        )

        height = float(
            size.findtext("height", "0")
        )

        if width <= 0 or height <= 0:
            return []

        for obj in root.findall("object"):

            name = obj.findtext(
                "name",
                ""
            )

            biological_class = normalize_class(
                name
            )

            if biological_class is None:
                continue

            box = obj.find("bndbox")

            if box is None:
                continue

            try:

                xmin = float(
                    box.findtext("xmin", "0")
                )

                ymin = float(
                    box.findtext("ymin", "0")
                )

                xmax = float(
                    box.findtext("xmax", "0")
                )

                ymax = float(
                    box.findtext("ymax", "0")
                )

            except ValueError:
                continue

            result = yolo_box(
                xmin,
                ymin,
                xmax,
                ymax,
                width,
                height
            )

            if result is None:
                continue

            annotations.append(
                (
                    CLASS_MAP[biological_class],
                    *result
                )
            )

        return annotations

    except Exception:
        return []


# ============================================================
# YOLO TXT
# ============================================================

def parse_yolo_txt(annotation_file):

    annotations = []

    try:

        with open(
            annotation_file,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:

                    class_id = int(parts[0])

                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:
                    continue

                if class_id not in {
                    0, 1, 2, 3
                }:
                    continue

                if not (
                    0 <= xc <= 1
                    and 0 <= yc <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    continue

                annotations.append(
                    (
                        class_id,
                        xc,
                        yc,
                        w,
                        h
                    )
                )

        return annotations

    except Exception:
        return []


# ============================================================
# WRITE YOLO
# ============================================================

def write_yolo(
    output_file,
    annotations
):

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for class_id, xc, yc, w, h in annotations:

            f.write(
                f"{class_id} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


# ============================================================
# BUILD IMAGE INDEX
# ============================================================

def build_image_index():

    print("\nBuilding image index...")

    image_index = defaultdict(list)

    total_images = 0

    for root, dirs, files in os.walk(
        SOURCE_ROOT
    ):

        # Ignore Mac metadata
        dirs[:] = [
            d for d in dirs
            if d != "__MACOSX"
        ]

        for file in files:

            extension = os.path.splitext(
                file
            )[1].lower()

            if extension not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp"
            }:
                continue

            full_path = os.path.join(
                root,
                file
            )

            stem = os.path.splitext(
                file
            )[0].lower()

            image_index[stem].append(
                full_path
            )

            total_images += 1

    print(
        f"Indexed {total_images:,} images."
    )

    return image_index


# ============================================================
# FIND IMAGE FROM INDEX
# ============================================================

def find_image_fast(
    annotation_file,
    image_index
):

    stem = os.path.splitext(
        os.path.basename(
            annotation_file
        )
    )[0].lower()

    candidates = image_index.get(
        stem,
        []
    )

    if not candidates:
        return None

    # Prefer image in same directory
    annotation_dir = os.path.dirname(
        annotation_file
    )

    for image in candidates:

        if os.path.dirname(image) == annotation_dir:
            return image

    # Otherwise return first match
    return candidates[0]


# ============================================================
# PROCESS ONE ANNOTATION
# ============================================================

def process_annotation(
    annotation_file,
    image_index
):

    extension = os.path.splitext(
        annotation_file
    )[1].lower()

    image_file = find_image_fast(
        annotation_file,
        image_index
    )

    if image_file is None:

        return {
            "success": False,
            "reason": "NO_IMAGE",
            "annotation_file": annotation_file
        }

    # --------------------------------------------------------
    # Determine format
    # --------------------------------------------------------

    if extension == ".xml":

        try:

            with open(
                annotation_file,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                beginning = f.read(3000)

        except Exception:

            beginning = ""

        if "<bndbox>" not in beginning:

            return {
                "success": False,
                "reason": "UNSUPPORTED_XML",
                "annotation_file": annotation_file
            }

        annotations = parse_voc_xml(
            annotation_file
        )

        format_name = "VOC"

    else:

        # Quickly determine whether TXT is YOLO
        is_yolo = False

        try:

            with open(
                annotation_file,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                first_line = ""

                for line in f:

                    if line.strip():

                        first_line = line.strip()
                        break

            parts = first_line.split()

            if len(parts) == 5:

                try:

                    class_id = int(parts[0])

                    values = [
                        float(x)
                        for x in parts[1:]
                    ]

                    if (
                        class_id in {0, 1, 2, 3}
                        and
                        all(
                            0 <= x <= 1
                            for x in values
                        )
                    ):
                        is_yolo = True

                except ValueError:
                    pass

        except Exception:

            pass

        if is_yolo:

            annotations = parse_yolo_txt(
                annotation_file
            )

            format_name = "YOLO"

        else:

            annotations = parse_nih_txt(
                annotation_file
            )

            format_name = "NIH_TXT"

    # --------------------------------------------------------
    # No usable annotation
    # --------------------------------------------------------

    if not annotations:

        return {
            "success": False,
            "reason": "NO_USABLE_LABEL",
            "annotation_file": annotation_file,
            "image_file": image_file
        }

    # --------------------------------------------------------
    # Unique output name
    # --------------------------------------------------------

    relative_image = os.path.relpath(
        image_file,
        SOURCE_ROOT
    )

    safe_name = os.path.splitext(
        relative_image
    )[0]

    safe_name = safe_name.replace(
        "\\",
        "__"
    ).replace(
        "/",
        "__"
    )

    output_file = os.path.join(
        LABEL_ROOT,
        safe_name + ".txt"
    )

    write_yolo(
        output_file,
        annotations
    )

    return {
        "success": True,
        "image_file": image_file,
        "label_file": output_file,
        "annotation_file": annotation_file,
        "format": format_name,
        "objects": len(annotations),
        "classes": [
            x[0]
            for x in annotations
        ]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 3 - FAST ANNOTATION CONVERSION TO YOLO")
    print("=" * 75)

    print("\nSource:")
    print(SOURCE_ROOT)

    print("\nOutput:")
    print(LABEL_ROOT)

    if not os.path.isdir(SOURCE_ROOT):

        print("\nERROR:")
        print("Dataset folder not found.")

        return

    os.makedirs(
        LABEL_ROOT,
        exist_ok=True
    )

    # ========================================================
    # 1. Build image index ONCE
    # ========================================================

    image_index = build_image_index()

    # ========================================================
    # 2. Find annotations
    # ========================================================

    print("\nFinding annotation files...")

    annotation_files = []

    for root, dirs, files in os.walk(
        SOURCE_ROOT
    ):

        dirs[:] = [
            d for d in dirs
            if d != "__MACOSX"
        ]

        for file in files:

            extension = os.path.splitext(
                file
            )[1].lower()

            if extension not in {
                ".txt",
                ".xml"
            }:
                continue

            full_path = os.path.join(
                root,
                file
            )

            if "__MACOSX" in full_path:
                continue

            annotation_files.append(
                full_path
            )

    print(
        f"Annotation files found: "
        f"{len(annotation_files):,}"
    )

    # ========================================================
    # 3. Parallel processing
    # ========================================================

    results = []

    converted = 0
    no_image = 0
    no_label = 0
    unsupported = 0

    format_counts = Counter()
    class_counts = Counter()

    total = len(annotation_files)

    print(
        f"\nProcessing with "
        f"{WORKERS} workers..."
    )

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_annotation,
                annotation_file,
                image_index
            )
            for annotation_file
            in annotation_files
        ]

        for index, future in enumerate(
            as_completed(futures),
            1
        ):

            try:

                result = future.result()

            except Exception as e:

                result = {
                    "success": False,
                    "reason": "ERROR",
                    "error": str(e)
                }

            results.append(result)

            if result.get("success"):

                converted += 1

                format_counts[
                    result["format"]
                ] += 1

                for class_id in result.get(
                    "classes",
                    []
                ):

                    class_counts[
                        class_id
                    ] += 1

            else:

                reason = result.get(
                    "reason",
                    ""
                )

                if reason == "NO_IMAGE":
                    no_image += 1

                elif reason == "NO_USABLE_LABEL":
                    no_label += 1

                else:
                    unsupported += 1

            if (
                index % 500 == 0
                or index == total
            ):

                print(
                    f"Processed "
                    f"{index:,}/{total:,}"
                )

    # ========================================================
    # 4. Save manifest
    # ========================================================

    successful = [
        x for x in results
        if x.get("success")
    ]

    os.makedirs(
        os.path.dirname(
            OUTPUT_MANIFEST
        ),
        exist_ok=True
    )

    with open(
        OUTPUT_MANIFEST,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_file",
                "label_file",
                "annotation_source",
                "format",
                "objects"
            ]
        )

        writer.writeheader()

        for result in successful:

            writer.writerow({

                "image_file":
                    result["image_file"],

                "label_file":
                    result["label_file"],

                "annotation_source":
                    result["annotation_file"],

                "format":
                    result["format"],

                "objects":
                    result["objects"]
            })

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("CONVERSION SUMMARY")
    print("=" * 75)

    print(
        "\nAnnotation files:",
        f"{total:,}"
    )

    print(
        "Successfully converted:",
        f"{converted:,}"
    )

    print(
        "No matching image:",
        f"{no_image:,}"
    )

    print(
        "No usable annotation:",
        f"{no_label:,}"
    )

    print(
        "Unsupported / errors:",
        f"{unsupported:,}"
    )

    print("\nFORMAT DISTRIBUTION")

    for name, count in format_counts.most_common():

        print(
            f"{name:15s} : {count:,}"
        )

    print("\nCLASS DISTRIBUTION")

    names = {
        0: "RBC",
        1: "WBC",
        2: "Platelet",
        3: "Malaria_Parasite"
    }

    for class_id in range(4):

        print(
            f"{class_id} = "
            f"{names[class_id]:18s} : "
            f"{class_counts[class_id]:,}"
        )

    print("\nOUTPUT")

    print(
        LABEL_ROOT
    )

    print(
        OUTPUT_MANIFEST
    )

    print("\n" + "=" * 75)
    print("STEP 3 COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()