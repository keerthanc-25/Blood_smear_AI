from pathlib import Path
import csv
import hashlib
import shutil
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

SOURCE_DIR = Path(r"D:\blood smear images")
VERIFIED_DATASET = Path(r"D:\Blood_Smear_AI\data\yolo_dataset_3class")
CURRENT_OUTPUT = Path(r"D:\Blood_Smear_AI\data\pseudo_labeled_new")
SUMMARY_CSV = CURRENT_OUTPUT / "pseudo_label_summary.csv"
OUTPUT_DIR = Path(r"D:\Blood_Smear_AI\data\rbc_second_pass")
MODEL_PATH = Path(r"D:\Blood_Smear_AI\runs\blood_smear_rbc_wbc_platelet_finetuned\weights\best.pt")

RBC_CONF_THRESHOLD = 0.50
IMGSZ = 800
BATCH_SIZE = 2
DEVICE = 0

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


def verified_hashes():
    hashes = set()
    images = find_images(VERIFIED_DATASET / "images")
    print(f"Verified images: {len(images)}")
    for i, p in enumerate(images, 1):
        try:
            hashes.add(get_hash(p))
        except Exception:
            pass
        if i % 500 == 0:
            print(f"  Hashed {i}/{len(images)}")
    return hashes


def previous_paths():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Missing first-pass CSV: {SUMMARY_CSV}"
        )

    processed = set()
    wp = []

    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = Path(row["source_image"])
            processed.add(str(path))
            if row["group"] == "wbc_platelet":
                wp.append(path)

    print(f"First-pass processed: {len(processed)}")
    print(f"WBC/Platelet images to rescan for RBC: {len(wp)}")
    return processed, wp


def untouched_images(processed, verified):
    all_images = find_images(SOURCE_DIR)
    untouched = []
    seen = set()
    invalid = 0

    for i, p in enumerate(all_images, 1):
        if str(p) in processed:
            continue

        if not valid_image(p):
            invalid += 1
            continue

        try:
            h = get_hash(p)
        except Exception:
            invalid += 1
            continue

        if h in seen or h in verified:
            continue

        seen.add(h)
        untouched.append(p)

        if i % 2000 == 0:
            print(f"Scanned {i}/{len(all_images)} | untouched: {len(untouched)}")

    print(f"Untouched images: {len(untouched)}")
    print(f"Invalid skipped: {invalid}")
    return untouched


def save_label(path, detections):
    with open(path, "w", encoding="utf-8") as f:
        for x, y, w, h in detections:
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def process(model, images):
    out_i = OUTPUT_DIR / "images"
    out_l = OUTPUT_DIR / "labels"
    out_i.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)

    accepted = 0
    objects = 0
    errors = 0
    rows = []

    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start:start + BATCH_SIZE]
        print(
            f"\nRBC scan {start + 1}-"
            f"{min(start + BATCH_SIZE, len(images))}/{len(images)}"
        )

        try:
            results = model.predict(
                source=[str(x) for x in batch],
                conf=RBC_CONF_THRESHOLD,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False
            )
        except Exception as e:
            print(f"[WARNING] Batch failed: {e}")
            results = []
            for image in batch:
                try:
                    r = model.predict(
                        source=str(image),
                        conf=RBC_CONF_THRESHOLD,
                        imgsz=IMGSZ,
                        device=DEVICE,
                        verbose=False
                    )
                    results.append(r[0] if r else None)
                except Exception as e2:
                    errors += 1
                    print(f"[SKIP] {image} -> {e2}")
                    results.append(None)

        for image, result in zip(batch, results):
            if result is None:
                rows.append([str(image), "PREDICTION_ERROR", 0, 0])
                continue

            detections = []
            confidences = []

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    if cls_id != 0 or conf <= RBC_CONF_THRESHOLD:
                        continue

                    x, y, w, h = box.xywhn[0].tolist()
                    detections.append((x, y, w, h))
                    confidences.append(conf)

            if not detections:
                rows.append([
                    str(image),
                    "NO_RBC_ABOVE_0.50",
                    0,
                    0
                ])
                continue

            try:
                name = f"{image.stem}_{get_hash(image)[:12]}{image.suffix.lower()}"
                shutil.copy2(image, out_i / name)
                save_label(out_l / f"{Path(name).stem}.txt", detections)
            except Exception as e:
                print(f"[SAVE ERROR] {image} -> {e}")
                rows.append([str(image), "SAVE_ERROR", 0, 0])
                continue

            accepted += 1
            objects += len(detections)
            rows.append([
                str(image),
                "RBC_ACCEPTED",
                len(detections),
                f"{max(confidences):.4f}"
            ])

        print(f"Accepted RBC images: {accepted}")
        print(f"RBC objects: {objects}")

    return rows, accepted, objects, errors


def main():
    print("=" * 70)
    print("RBC SECOND-PASS ANNOTATION")
    print("=" * 70)
    print("Threshold: RBC strictly > 0.50")
    print("Existing pseudo labels will NOT be modified.")

    print("\nLoading model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded.")

    processed, wp_paths = previous_paths()
    verified = verified_hashes()
    untouched = untouched_images(processed, verified)

    # WBC/Platelet images were already processed, but RBC was ignored.
    # Add them back to the pool, plus images untouched in pass 1.
    pool = [p for p in wp_paths if p.exists()]
    pool.extend(untouched)

    unique = []
    seen = set()

    for p in pool:
        try:
            h = get_hash(p)
        except Exception:
            continue
        if h not in seen:
            seen.add(h)
            unique.append(p)

    pool = unique

    print("\n" + "=" * 70)
    print("RBC SECOND-PASS PLAN")
    print("=" * 70)
    print(f"WBC/Platelet images rescanned for RBC: {len(wp_paths)}")
    print(f"Untouched images: {len(untouched)}")
    print(f"TOTAL RBC SECOND-PASS POOL: {len(pool)}")
    print("RBC threshold: > 0.50")

    if not pool:
        print("No images available for second pass.")
        return

    rows, accepted, objects, errors = process(model, pool)

    csv_path = OUTPUT_DIR / "rbc_second_pass_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_image", "status", "RBC_objects", "max_RBC_conf"
        ])
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("RBC SECOND-PASS COMPLETE")
    print("=" * 70)
    print(f"Images scanned:      {len(pool)}")
    print(f"RBC images accepted: {accepted}")
    print(f"RBC objects found:   {objects}")
    print(f"Prediction errors:   {errors}")
    print("\nPrevious first-pass RBC: 418 images / 738 objects")
    print(f"Potential RBC total: {418 + accepted} images / {738 + objects} objects")
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"CSV:    {csv_path}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
