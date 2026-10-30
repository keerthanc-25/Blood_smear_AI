from pathlib import Path
import shutil
import hashlib
import random
import yaml
import cv2
import xml.etree.ElementTree as ET
import json
from collections import Counter, defaultdict

# ============================================================
# FINAL BLOOD SMEAR DATASET BUILDER - OPTIMIZED
# ============================================================

PROJECT = Path(r"D:\Blood_Smear_AI")

SOURCE_ROOT = Path(r"D:\blood smear images")

FINAL_ROOT = PROJECT / "data" / "final_dataset"

# ============================================================
# ALL DATA SOURCES
# ============================================================

SOURCES = [
    (
        "ORIGINAL",
        SOURCE_ROOT
    ),

    (
        "PSEUDO_NEW_RBC",
        PROJECT / "data" / "pseudo_labeled_new" / "rbc"
    ),

    (
        "PSEUDO_NEW_WBC_PLATELET",
        PROJECT / "data" / "pseudo_labeled_new" / "wbc_platelet"
    ),

    (
        "RBC_SECOND_PASS",
        PROJECT / "data" / "rbc_second_pass"
    ),

    (
        "RBC_TTA_SELECTED",
        PROJECT / "data" / "rbc_selected"
    ),

    (
        "RBC_TTA_CANDIDATES",
        PROJECT / "data" / "rbc_candidates"
    ),

    (
        "PSEUDO_ALL",
        PROJECT / "data" / "pseudo_labeled_3class_all"
    ),

    (
        "RBC_AUGMENTATION",
        PROJECT / "data" / "rbc_augmented"
    ),
]

# ============================================================
# FINAL CLASSES
# ============================================================

CLASS_NAMES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet",
}

# ============================================================
# SPLIT
# ============================================================

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

# ============================================================
# FILE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

ANNOTATION_EXTENSIONS = {
    ".txt",
    ".xml",
    ".json",
}


# ============================================================
# NORMALIZE CLASS
# ============================================================

def normalize_class_name(name):

    if name is None:
        return None

    n = str(name).strip().lower()

    n = n.replace("-", "_")
    n = n.replace(" ", "_")

    # RBC
    if n in {
        "rbc",
        "red_blood_cell",
        "red_blood_cells",
        "redbloodcell",
        "redbloodcells",
        "erythrocyte",
        "erythrocytes",
        "redcell",
        "redcells",
    }:
        return 0

    # WBC
    if n in {
        "wbc",
        "white_blood_cell",
        "white_blood_cells",
        "whitebloodcell",
        "whitebloodcells",
        "leukocyte",
        "leukocytes",
        "whitecell",
        "whitecells",
    }:
        return 1

    # Platelet
    if n in {
        "platelet",
        "platelets",
        "thrombocyte",
        "thrombocytes",
    }:
        return 2

    # Malaria / parasite
    if (
        "malaria" in n
        or "parasite" in n
        or "plasmodium" in n
        or "malarial" in n
    ):
        return None

    return None


# ============================================================
# IMAGE VALIDATION
# ============================================================

def valid_image(path):

    try:

        img = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return False

        if img.size == 0:
            return False

        return True

    except Exception:
        return False


# ============================================================
# SHA256
# ============================================================

def file_hash(path):

    try:

        h = hashlib.sha256()

        with open(
            path,
            "rb"
        ) as f:

            while True:

                chunk = f.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                h.update(chunk)

        return h.hexdigest()

    except Exception:
        return None


# ============================================================
# LOAD YAML CLASS MAP
# ============================================================

def load_yaml_class_names(yaml_path):

    if yaml_path is None:
        return None

    try:

        with open(
            yaml_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = yaml.safe_load(f)

        names = data.get("names")

        if isinstance(names, dict):

            result = {}

            for k, v in names.items():

                try:
                    result[int(k)] = v
                except:
                    pass

            return result

        if isinstance(names, list):

            return {
                i: value
                for i, value in enumerate(names)
            }

    except Exception:
        pass

    return None


# ============================================================
# BUILD CLASS MAP CACHE
# ============================================================

def build_yaml_cache(root):

    cache = {}

    print("\nIndexing YAML class maps...")

    yaml_files = []

    for p in root.rglob("*.yaml"):
        yaml_files.append(p)

    for p in root.rglob("*.yml"):
        yaml_files.append(p)

    for p in yaml_files:

        names = load_yaml_class_names(p)

        if names:

            cache[p.parent] = names

    print(
        "Class-map directories found:",
        len(cache)
    )

    return cache


# ============================================================
# FIND NEAREST CLASS MAP
# ============================================================

def get_class_map(annotation_path, yaml_cache):

    current = annotation_path.parent

    while True:

        if current in yaml_cache:
            return yaml_cache[current]

        if current.parent == current:
            break

        current = current.parent

    # Search ancestors only
    current = annotation_path.parent

    while current != current.parent:

        current = current.parent

        if current in yaml_cache:
            return yaml_cache[current]

    return None


# ============================================================
# CONVERT CLASS ID
# ============================================================

def convert_class_id(
    class_id,
    class_names=None
):

    try:
        cid = int(class_id)
    except:
        return None

    # Already final 3-class mapping
    if cid in CLASS_NAMES:
        return cid

    # Source class mapping
    if class_names:

        source_name = class_names.get(cid)

        if source_name is not None:

            return normalize_class_name(
                source_name
            )

    return None


# ============================================================
# READ YOLO TXT
# ============================================================

def read_yolo_label(
    path,
    class_names=None
):

    objects = []

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                try:

                    source_id = int(parts[0])

                    values = [
                        float(x)
                        for x in parts[1:5]
                    ]

                except:
                    continue

                cid = convert_class_id(
                    source_id,
                    class_names
                )

                if cid is None:
                    continue

                # Validate YOLO coordinates
                if not all(
                    0 <= x <= 1
                    for x in values
                ):
                    continue

                if values[2] <= 0:
                    continue

                if values[3] <= 0:
                    continue

                objects.append(
                    (
                        cid,
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                    )
                )

    except Exception:
        return []

    return objects


# ============================================================
# READ XML
# ============================================================

def read_xml_label(path):

    objects = []

    try:

        tree = ET.parse(path)

        root = tree.getroot()

        size = root.find("size")

        if size is None:
            return []

        width = float(
            size.findtext(
                "width",
                "0"
            )
        )

        height = float(
            size.findtext(
                "height",
                "0"
            )
        )

        if width <= 0 or height <= 0:
            return []

        for obj in root.findall("object"):

            name = obj.findtext(
                "name"
            )

            cid = normalize_class_name(
                name
            )

            # Ignore malaria
            if cid is None:
                continue

            box = obj.find("bndbox")

            if box is None:
                continue

            try:

                xmin = float(
                    box.findtext(
                        "xmin",
                        "0"
                    )
                )

                ymin = float(
                    box.findtext(
                        "ymin",
                        "0"
                    )
                )

                xmax = float(
                    box.findtext(
                        "xmax",
                        "0"
                    )
                )

                ymax = float(
                    box.findtext(
                        "ymax",
                        "0"
                    )
                )

            except:
                continue

            bw = xmax - xmin
            bh = ymax - ymin

            if bw <= 0 or bh <= 0:
                continue

            xc = (
                (xmin + xmax) / 2
            ) / width

            yc = (
                (ymin + ymax) / 2
            ) / height

            bw /= width
            bh /= height

            if not (
                0 <= xc <= 1
                and 0 <= yc <= 1
                and 0 < bw <= 1
                and 0 < bh <= 1
            ):
                continue

            objects.append(
                (
                    cid,
                    xc,
                    yc,
                    bw,
                    bh,
                )
            )

    except Exception:
        return []

    return objects


# ============================================================
# READ JSON
# ============================================================

def read_json_label(path):

    objects = []

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    # ---------------------------------------------
    # Generic object format
    # ---------------------------------------------

    candidates = []

    if isinstance(
        data.get("objects"),
        list
    ):
        candidates = data["objects"]

    elif isinstance(
        data.get("annotations"),
        list
    ):
        candidates = data["annotations"]

    for obj in candidates:

        if not isinstance(
            obj,
            dict
        ):
            continue

        name = (
            obj.get("class")
            or obj.get("label")
            or obj.get("name")
            or obj.get("category")
        )

        cid = normalize_class_name(
            name
        )

        if cid is None:
            continue

        bbox = (
            obj.get("bbox")
            or obj.get("box")
        )

        if bbox is None:
            continue

        try:

            x, y, w, h = map(
                float,
                bbox[:4]
            )

        except:
            continue

        # Assume normalized bbox
        if (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):

            objects.append(
                (
                    cid,
                    x,
                    y,
                    w,
                    h,
                )
            )

    return objects


# ============================================================
# READ ANNOTATION
# ============================================================

def read_annotation(
    annotation_path,
    yaml_cache
):

    if annotation_path is None:
        return []

    suffix = annotation_path.suffix.lower()

    if suffix == ".txt":

        class_names = get_class_map(
            annotation_path,
            yaml_cache
        )

        return read_yolo_label(
            annotation_path,
            class_names
        )

    if suffix == ".xml":

        return read_xml_label(
            annotation_path
        )

    if suffix == ".json":

        return read_json_label(
            annotation_path
        )

    return []


# ============================================================
# BUILD ANNOTATION INDEX
# ============================================================

def build_annotation_index(root):

    print("\n")
    print("=" * 80)
    print("BUILDING ANNOTATION INDEX")
    print("=" * 80)

    annotation_index = defaultdict(list)

    count = 0

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in ANNOTATION_EXTENSIONS:
            continue

        stem = path.stem.lower()

        annotation_index[stem].append(
            path
        )

        count += 1

        if count % 1000 == 0:

            print(
                f"Annotations indexed: "
                f"{count}"
            )

    print(
        "\nTotal annotation files indexed:",
        count
    )

    print(
        "Unique annotation names:",
        len(annotation_index)
    )

    return annotation_index


# ============================================================
# FIND ANNOTATION USING INDEX
# ============================================================

def find_indexed_annotation(
    image_path,
    annotation_index
):

    stem = image_path.stem.lower()

    candidates = annotation_index.get(
        stem,
        []
    )

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # --------------------------------------------------------
    # If multiple annotations have same filename,
    # prefer same parent directory.
    # --------------------------------------------------------

    for candidate in candidates:

        if candidate.parent == image_path.parent:

            return candidate

    # --------------------------------------------------------
    # Prefer annotation with parent name appearing
    # in image parent path.
    # --------------------------------------------------------

    image_parent_string = str(
        image_path.parent
    ).lower()

    for candidate in candidates:

        candidate_parent_string = str(
            candidate.parent
        ).lower()

        if (
            candidate.parent.name.lower()
            in image_parent_string
        ):

            return candidate

        if (
            image_path.parent.name.lower()
            in candidate_parent_string
        ):

            return candidate

    # Otherwise first match
    return candidates[0]


# ============================================================
# FIND ALL IMAGES
# ============================================================

def find_images(root):

    images = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in IMAGE_EXTENSIONS:

            images.append(path)

    return images


# ============================================================
# RECORD
# ============================================================

class Record:

    def __init__(
        self,
        image,
        label_text,
        source,
        image_hash
    ):

        self.image = image

        self.label_text = label_text

        self.source = source

        self.image_hash = image_hash

        classes = []

        for line in label_text.splitlines():

            parts = line.split()

            if not parts:
                continue

            try:

                cid = int(parts[0])

                if cid in CLASS_NAMES:
                    classes.append(cid)

            except:
                pass

        self.classes = set(classes)

        self.signature = tuple(
            sorted(
                self.classes
            )
        )


# ============================================================
# OBJECTS → YOLO
# ============================================================

def objects_to_yolo(objects):

    lines = []

    for obj in objects:

        if len(obj) < 5:
            continue

        cid = int(obj[0])

        if cid not in CLASS_NAMES:
            continue

        xc = float(obj[1])
        yc = float(obj[2])
        w = float(obj[3])
        h = float(obj[4])

        if not (
            0 <= xc <= 1
            and 0 <= yc <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            continue

        lines.append(
            f"{cid} "
            f"{xc:.6f} "
            f"{yc:.6f} "
            f"{w:.6f} "
            f"{h:.6f}"
        )

    return lines


# ============================================================
# PROCESS ORIGINAL DATASET
# ============================================================

def process_original(
    annotation_index,
    yaml_cache
):

    print("\n")
    print("=" * 80)
    print("PROCESSING ORIGINAL DATA")
    print("=" * 80)

    images = find_images(
        SOURCE_ROOT
    )

    total = len(images)

    print(
        f"Original images discovered: {total}"
    )

    records = []

    seen_hashes = set()

    no_annotation = 0
    invalid = 0
    empty_annotation = 0
    duplicates = 0

    for i, image_path in enumerate(
        images,
        start=1
    ):

        # Progress every 1000
        if (
            i % 1000 == 0
            or i == 1
            or i == total
        ):

            print(
                f"Original progress: "
                f"{i:,} / {total:,}"
            )

        # ----------------------------------------
        # Annotation lookup
        # ----------------------------------------

        annotation = find_indexed_annotation(
            image_path,
            annotation_index
        )

        if annotation is None:

            no_annotation += 1

            continue

        # ----------------------------------------
        # Read annotation first
        # ----------------------------------------

        objects = read_annotation(
            annotation,
            yaml_cache
        )

        if not objects:

            empty_annotation += 1

            continue

        label_lines = objects_to_yolo(
            objects
        )

        if not label_lines:

            empty_annotation += 1

            continue

        # ----------------------------------------
        # Validate image
        # ----------------------------------------

        if not valid_image(
            image_path
        ):

            invalid += 1

            continue

        # ----------------------------------------
        # Hash
        # ----------------------------------------

        img_hash = file_hash(
            image_path
        )

        if img_hash is None:

            invalid += 1

            continue

        if img_hash in seen_hashes:

            duplicates += 1

            continue

        seen_hashes.add(
            img_hash
        )

        # ----------------------------------------
        # Create record
        # ----------------------------------------

        label_text = (
            "\n".join(
                label_lines
            )
            + "\n"
        )

        records.append(
            Record(
                image_path,
                label_text,
                "ORIGINAL",
                img_hash
            )
        )

    print("\nOriginal processing complete.")

    print(
        "Accepted:",
        len(records)
    )

    print(
        "No annotation:",
        no_annotation
    )

    print(
        "Invalid:",
        invalid
    )

    print(
        "Empty/unsupported annotation:",
        empty_annotation
    )

    print(
        "Duplicates:",
        duplicates
    )

    return records, seen_hashes


# ============================================================
# PROCESS YOLO-STYLE DATASET
# ============================================================

def process_yolo_source(
    source_name,
    source_root,
    global_hashes
):

    print("\n")
    print("-" * 80)
    print(source_name)
    print(source_root)
    print("-" * 80)

    if not source_root.exists():

        print("NOT FOUND - SKIPPED")

        return []

    # --------------------------------------------------------
    # Look for images/labels
    # --------------------------------------------------------

    image_root = (
        source_root / "images"
    )

    label_root = (
        source_root / "labels"
    )

    if not image_root.exists():

        image_root = source_root

    if not label_root.exists():

        # If no labels directory,
        # use annotation index local to source.
        label_root = source_root

    images = find_images(
        image_root
    )

    print(
        "Images discovered:",
        len(images)
    )

    records = []

    local_seen = set()

    skipped_no_label = 0
    skipped_invalid = 0
    skipped_duplicate = 0

    # Build direct label index
    label_index = {}

    for p in label_root.rglob("*.txt"):

        label_index[
            p.stem.lower()
        ] = p

    for i, image_path in enumerate(
        images,
        start=1
    ):

        if (
            i % 1000 == 0
            or i == len(images)
        ):

            print(
                f"Progress: "
                f"{i:,} / {len(images):,}"
            )

        label_path = label_index.get(
            image_path.stem.lower()
        )

        if label_path is None:

            skipped_no_label += 1

            continue

        objects = read_yolo_label(
            label_path,
            None
        )

        if not objects:

            continue

        label_lines = objects_to_yolo(
            objects
        )

        if not label_lines:

            continue

        if not valid_image(
            image_path
        ):

            skipped_invalid += 1

            continue

        img_hash = file_hash(
            image_path
        )

        if img_hash is None:

            skipped_invalid += 1

            continue

        if img_hash in global_hashes:

            skipped_duplicate += 1

            continue

        if img_hash in local_seen:

            skipped_duplicate += 1

            continue

        local_seen.add(
            img_hash
        )

        global_hashes.add(
            img_hash
        )

        label_text = (
            "\n".join(
                label_lines
            )
            + "\n"
        )

        records.append(
            Record(
                image_path,
                label_text,
                source_name,
                img_hash
            )
        )

    print(
        "Accepted:",
        len(records)
    )

    print(
        "No label:",
        skipped_no_label
    )

    print(
        "Invalid:",
        skipped_invalid
    )

    print(
        "Duplicates:",
        skipped_duplicate
    )

    return records


# ============================================================
# PREPARE FINAL DIRECTORY
# ============================================================

def prepare_output():

    if FINAL_ROOT.exists():

        print(
            "\nRemoving previous final dataset..."
        )

        shutil.rmtree(
            FINAL_ROOT
        )

    for split in [
        "train",
        "val",
        "test",
    ]:

        (
            FINAL_ROOT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            FINAL_ROOT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# SPLIT
# ============================================================

def split_records(records):

    print("\n")
    print("=" * 80)
    print("CREATING 80 / 10 / 10 SPLIT")
    print("=" * 80)

    random.seed(
        SEED
    )

    groups = defaultdict(list)

    for record in records:

        groups[
            record.signature
        ].append(record)

    train = []
    val = []
    test = []

    for signature, group in groups.items():

        random.shuffle(group)

        n = len(group)

        n_train = int(
            n * TRAIN_RATIO
        )

        n_val = int(
            n * VAL_RATIO
        )

        train.extend(
            group[
                :n_train
            ]
        )

        val.extend(
            group[
                n_train:
                n_train + n_val
            ]
        )

        test.extend(
            group[
                n_train + n_val:
            ]
        )

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(
        f"\nTRAIN      : {len(train):,}"
    )

    print(
        f"VALIDATION : {len(val):,}"
    )

    print(
        f"TEST       : {len(test):,}"
    )

    print(
        f"TOTAL      : "
        f"{len(train) + len(val) + len(test):,}"
    )

    return train, val, test


# ============================================================
# WRITE DATA
# ============================================================

def write_split(
    records,
    split
):

    image_dir = (
        FINAL_ROOT
        / "images"
        / split
    )

    label_dir = (
        FINAL_ROOT
        / "labels"
        / split
    )

    total = len(records)

    for i, record in enumerate(
        records,
        start=1
    ):

        if (
            i % 1000 == 0
            or i == total
        ):

            print(
                f"Writing {split}: "
                f"{i:,} / {total:,}"
            )

        suffix = (
            record.image.suffix.lower()
        )

        filename = (
            f"{split}_{i:06d}"
        )

        image_out = (
            image_dir
            / f"{filename}{suffix}"
        )

        label_out = (
            label_dir
            / f"{filename}.txt"
        )

        try:

            shutil.copy2(
                record.image,
                image_out
            )

            with open(
                label_out,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    record.label_text
                )

        except Exception as e:

            print(
                "\nCOPY ERROR:",
                record.image
            )

            print(e)


# ============================================================
# OBJECT COUNTS
# ============================================================

def count_objects(records):

    counts = Counter()

    for record in records:

        for line in (
            record.label_text
            .splitlines()
        ):

            parts = line.split()

            if not parts:
                continue

            try:

                cid = int(parts[0])

                if cid in CLASS_NAMES:

                    counts[cid] += 1

            except:
                pass

    return counts


# ============================================================
# SOURCE COUNTS
# ============================================================

def source_summary(records):

    counts = Counter()

    for record in records:

        counts[
            record.source
        ] += 1

    return counts


# ============================================================
# WRITE YAML
# ============================================================

def write_yaml():

    data = {

        "path": str(
            FINAL_ROOT
        ).replace(
            "\\",
            "/"
        ),

        "train": "images/train",

        "val": "images/val",

        "test": "images/test",

        "names": {
            0: "RBC",
            1: "WBC",
            2: "Platelet",
        },
    }

    yaml_path = (
        FINAL_ROOT
        / "data.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            data,
            f,
            sort_keys=False
        )

    return yaml_path


# ============================================================
# FINAL REPORT
# ============================================================

def final_report(
    records,
    train,
    val,
    test
):

    train_counts = count_objects(
        train
    )

    val_counts = count_objects(
        val
    )

    test_counts = count_objects(
        test
    )

    sources = source_summary(
        records
    )

    print("\n")
    print("=" * 80)
    print("FINAL DATASET COMPLETE")
    print("=" * 80)

    print("\nSOURCE IMAGE COUNTS")
    print("-" * 60)

    for source, count in sources.items():

        print(
            f"{source:<30} : {count:,}"
        )

    print("\n")
    print("=" * 80)

    print("IMAGE COUNTS")
    print("-" * 60)

    print(
        f"TOTAL UNIQUE IMAGES : "
        f"{len(records):,}"
    )

    print(
        f"FINAL TRAIN         : "
        f"{len(train):,}"
    )

    print(
        f"FINAL VALIDATION    : "
        f"{len(val):,}"
    )

    print(
        f"FINAL TEST          : "
        f"{len(test):,}"
    )

    print(
        f"FINAL TOTAL         : "
        f"{len(train) + len(val) + len(test):,}"
    )

    print("\nOBJECT COUNTS")
    print("-" * 70)

    print(
        f"{'CLASS':<15}"
        f"{'TRAIN':>12}"
        f"{'VAL':>12}"
        f"{'TEST':>12}"
        f"{'TOTAL':>12}"
    )

    print("-" * 70)

    for cid, name in CLASS_NAMES.items():

        tr = train_counts[cid]
        va = val_counts[cid]
        te = test_counts[cid]

        print(
            f"{name:<15}"
            f"{tr:>12,}"
            f"{va:>12,}"
            f"{te:>12,}"
            f"{tr + va + te:>12,}"
        )

    print("\n")
    print("=" * 80)
    print("FINAL DATASET RULES")
    print("=" * 80)

    print(
        "Original annotated       -> INCLUDED"
    )

    print(
        "Pseudo labelled          -> INCLUDED"
    )

    print(
        "RBC second pass          -> INCLUDED"
    )

    print(
        "RBC TTA selected         -> INCLUDED"
    )

    print(
        "RBC TTA candidates       -> INCLUDED"
    )

    print(
        "RBC augmentation         -> INCLUDED"
    )

    print(
        "Malaria / parasite       -> EXCLUDED"
    )

    print(
        "Exact duplicate images   -> REMOVED"
    )

    print(
        "Train / Validation/Test  -> 80 / 10 / 10"
    )

    print(
        "Original source files    -> NOT MODIFIED"
    )

    print("\n")
    print(
        "FINAL DATASET:"
    )

    print(
        FINAL_ROOT
    )

    print("\nDATA YAML:")

    print(
        FINAL_ROOT / "data.yaml"
    )

    print("\n")
    print("=" * 80)
    print("READY FOR NEXT STAGE")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("FINAL BLOOD SMEAR DATASET BUILDER")
    print("OPTIMIZED VERSION")
    print("=" * 80)

    print(
        "\nSources:"
    )

    for name, path in SOURCES:

        print(
            f"  {name:<28} -> {path}"
        )

    print(
        "\nClasses:"
    )

    print(
        "  0 = RBC"
    )

    print(
        "  1 = WBC"
    )

    print(
        "  2 = Platelet"
    )

    print(
        "\nSplit:"
    )

    print(
        "  80% Train"
    )

    print(
        "  10% Validation"
    )

    print(
        "  10% Test"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    prepare_output()

    # --------------------------------------------------------
    # Build annotation index ONCE
    # --------------------------------------------------------

    annotation_index = build_annotation_index(
        SOURCE_ROOT
    )

    # --------------------------------------------------------
    # YAML cache
    # --------------------------------------------------------

    yaml_cache = build_yaml_cache(
        SOURCE_ROOT
    )

    # --------------------------------------------------------
    # Original data
    # --------------------------------------------------------

    all_records, global_hashes = (
        process_original(
            annotation_index,
            yaml_cache
        )
    )

    # --------------------------------------------------------
    # Other datasets
    # --------------------------------------------------------

    for source_name, source_root in SOURCES:

        if source_name == "ORIGINAL":
            continue

        new_records = process_yolo_source(
            source_name,
            source_root,
            global_hashes
        )

        all_records.extend(
            new_records
        )

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if not all_records:

        print(
            "\nERROR:"
        )

        print(
            "No usable data was found."
        )

        return

    print("\n")
    print("=" * 80)
    print("COMBINED DATASET")
    print("=" * 80)

    print(
        f"Unique usable images: "
        f"{len(all_records):,}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, val, test = split_records(
        all_records
    )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("WRITING FINAL DATASET")
    print("=" * 80)

    write_split(
        train,
        "train"
    )

    write_split(
        val,
        "val"
    )

    write_split(
        test,
        "test"
    )

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    write_yaml()

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    final_report(
        all_records,
        train,
        val,
        test
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()