from pathlib import Path
import csv
import hashlib
import shutil
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import numpy as np

# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = Path(r"D:\blood smear images")
VERIFIED_DATASET = Path(r"D:\Blood_Smear_AI\data\yolo_dataset_3class")

FIRST_PASS = Path(r"D:\Blood_Smear_AI\data\pseudo_labeled_new")
SECOND_PASS = Path(r"D:\Blood_Smear_AI\data\rbc_second_pass")

OUTPUT_DIR = Path(r"D:\Blood_Smear_AI\data\rbc_candidates")

MODEL_PATH = Path(
    r"D:\Blood_Smear_AI\runs\blood_smear_rbc_wbc_platelet_finetuned"
    r"\weights\best.pt"
)

# Candidate search threshold. These are NOT automatically trusted labels.
CANDIDATE_CONF = 0.20

# Final candidate requirements
MIN_TTA_VIEWS = 2
MIN_AVG_CONF = 0.30
MIN_IOU = 0.50

IMGSZ = 800
DEVICE = 0
BATCH_SIZE = 1   # safer for GTX 1050 Ti with 3 TTA views

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# HELPERS
# ============================================================

def get_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return True
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False


def find_images(folder):
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTS
        and "_MACOSX" not in p.parts
    ]


def iou(a, b):
    # xyxy normalized
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(x, y, w, h):
    return [
        max(0.0, x - w / 2),
        max(0.0, y - h / 2),
        min(1.0, x + w / 2),
        min(1.0, y + h / 2),
    ]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        x2 - x1,
        y2 - y1,
    )


def transform_back(box, mode):
    x1, y1, x2, y2 = box

    if mode == "hflip":
        return [1 - x2, y1, 1 - x1, y2]

    if mode == "vflip":
        return [x1, 1 - y2, x2, 1 - y1]

    return box


def predict_rbc(model, image_path, mode):
    """
    Returns [(box_xyxy_normalized, confidence), ...]
    Only RBC class 0 is retained.
    """
    if mode == "original":
        source = str(image_path)
    else:
        with Image.open(image_path) as im:
            im = im.convert("RGB")

            if mode == "hflip":
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            else:
                im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            source = np.array(im)

    result = model.predict(
        source=source,
        conf=CANDIDATE_CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False
    )[0]

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        if cls_id != 0 or conf < CANDIDATE_CONF:
            continue

        x, y, w, h = box.xywhn[0].tolist()

        box_xyxy = xywh_to_xyxy(x, y, w, h)
        box_xyxy = transform_back(box_xyxy, mode)

        detections.append((box_xyxy, conf))

    return detections


def get_accepted_hashes():
    """
    Exclude verified images and already accepted RBC images.
    """
    hashes = set()

    for folder in [
        VERIFIED_DATASET / "images",
        FIRST_PASS / "rbc" / "images",
        SECOND_PASS / "images",
    ]:
        for image in find_images(folder):
            try:
                hashes.add(get_hash(image))
            except Exception:
                pass

    return hashes


def make_candidate(detections_by_view):
    """
    Require one RBC box to be supported by at least 2 TTA views.
    Candidate confidence = average confidence of matched views.
    """
    candidates = []

    original = detections_by_view["original"]

    for base_box, base_conf in original:

        matches = [(base_box, base_conf, "original")]

        for mode in ["hflip", "vflip"]:
            best_iou = 0.0
            best = None

            for box, conf in detections_by_view[mode]:
                score = iou(base_box, box)
                if score > best_iou:
                    best_iou = score
                    best = (box, conf)

            if best is not None and best_iou >= MIN_IOU:
                matches.append((best[0], best[1], mode))

        if len(matches) < MIN_TTA_VIEWS:
            continue

        avg_conf = sum(x[1] for x in matches) / len(matches)

        if avg_conf < MIN_AVG_CONF:
            continue

        # Average matched boxes after mapping them back to original coordinates.
        x1 = sum(x[0][0] for x in matches) / len(matches)
        y1 = sum(x[0][1] for x in matches) / len(matches)
        x2 = sum(x[0][2] for x in matches) / len(matches)
        y2 = sum(x[0][3] for x in matches) / len(matches)

        candidates.append({
            "box": [x1, y1, x2, y2],
            "avg_conf": avg_conf,
            "views": len(matches),
            "max_conf": max(x[1] for x in matches),
        })

    return candidates


def save_label(path, candidates):
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            x, y, w, h = xyxy_to_xywh(c["box"])
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("RBC CANDIDATE MINING + TTA")
    print("=" * 70)

    print(f"Candidate search confidence: >= {CANDIDATE_CONF}")
    print(f"Minimum TTA views: {MIN_TTA_VIEWS}/3")
    print(f"Minimum average confidence: {MIN_AVG_CONF}")
    print(f"Minimum box IoU agreement: {MIN_IOU}")

    OUTPUT_DIR.joinpath("images").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.joinpath("labels").mkdir(parents=True, exist_ok=True)

    print("\nLoading model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded.")

    print("\nCollecting already accepted image hashes...")
    accepted_hashes = get_accepted_hashes()
    print(f"Images excluded from candidate mining: {len(accepted_hashes)}")

    print("\nScanning source...")
    all_images = find_images(SOURCE_DIR)

    candidates_pool = []
    seen = set()
    invalid = 0

    for i, image in enumerate(all_images, 1):
        if not valid_image(image):
            invalid += 1
            continue

        try:
            h = get_hash(image)
        except Exception:
            invalid += 1
            continue

        if h in seen or h in accepted_hashes:
            continue

        seen.add(h)
        candidates_pool.append(image)

        if i % 5000 == 0:
            print(
                f"Scanned {i}/{len(all_images)} | "
                f"candidate pool: {len(candidates_pool)}"
            )

    print("\n" + "=" * 70)
    print("CANDIDATE POOL")
    print("=" * 70)
    print(f"Source images: {len(all_images)}")
    print(f"Images excluded: {len(accepted_hashes)}")
    print(f"Invalid skipped: {invalid}")
    print(f"Images to TTA scan: {len(candidates_pool)}")

    rows = []
    accepted_images = 0
    accepted_objects = 0
    prediction_errors = 0

    for index, image in enumerate(candidates_pool, 1):
        print(
            f"\nTTA {index}/{len(candidates_pool)}: "
            f"{image.name}"
        )

        try:
            detections = {}

            for mode in ["original", "hflip", "vflip"]:
                detections[mode] = predict_rbc(
                    model,
                    image,
                    mode
                )

            candidates = make_candidate(detections)

        except Exception as e:
            prediction_errors += 1
            print(f"[SKIP ERROR] {image} -> {e}")

            rows.append([
                str(image),
                "PREDICTION_ERROR",
                0,
                0,
                0,
            ])
            continue

        if not candidates:
            rows.append([
                str(image),
                "REJECTED_TTA",
                0,
                0,
                0,
            ])
            continue

        try:
            name = (
                f"{image.stem}_"
                f"{get_hash(image)[:12]}"
                f"{image.suffix.lower()}"
            )

            shutil.copy2(
                image,
                OUTPUT_DIR / "images" / name
            )

            save_label(
                OUTPUT_DIR / "labels" /
                f"{Path(name).stem}.txt",
                candidates
            )

        except Exception as e:
            print(f"[SAVE ERROR] {image} -> {e}")
            rows.append([
                str(image),
                "SAVE_ERROR",
                0,
                0,
                0,
            ])
            continue

        accepted_images += 1
        accepted_objects += len(candidates)

        max_conf = max(c["max_conf"] for c in candidates)
        avg_conf = sum(c["avg_conf"] for c in candidates) / len(candidates)
        min_views = min(c["views"] for c in candidates)

        rows.append([
            str(image),
            "TTA_ACCEPTED",
            len(candidates),
            f"{max_conf:.4f}",
            f"{avg_conf:.4f}",
            min_views,
        ])

        print(
            f"ACCEPTED | RBC objects: {len(candidates)} | "
            f"avg conf: {avg_conf:.3f} | "
            f"views: {min_views}"
        )

    csv_path = OUTPUT_DIR / "rbc_candidate_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_image",
            "status",
            "RBC_objects",
            "max_conf",
            "avg_conf",
            "min_tta_views",
        ])
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("RBC CANDIDATE TTA COMPLETE")
    print("=" * 70)
    print(f"Images scanned:       {len(candidates_pool)}")
    print(f"Candidate images:     {accepted_images}")
    print(f"Candidate RBC objects:{accepted_objects}")
    print(f"Prediction errors:    {prediction_errors}")
    print(f"Output:               {OUTPUT_DIR}")
    print(f"CSV:                  {csv_path}")
    print("\nThese are CANDIDATES, not yet trusted training labels.")
    print("DONE.")


if __name__ == "__main__":
    main()
