from pathlib import Path
from PIL import Image
import shutil


ROOT = Path(r"D:\Blood_Smear_AI\data\final_dataset")

REPORT = ROOT / "tiff_problem_files.txt"

QUARANTINE = ROOT / "_tiff_quarantine"


print("=" * 70)
print("REPAIRING PROBLEMATIC MULTI-PAGE TIFF FILES")
print("=" * 70)


if not REPORT.exists():
    raise FileNotFoundError(
        f"Problem report not found:\n{REPORT}"
    )


# ------------------------------------------------------------
# Read problem report
# ------------------------------------------------------------

problem_files = []

with open(REPORT, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        parts = line.split(" | ")

        if len(parts) >= 2:

            split = parts[0]
            image_path = Path(parts[1])

            problem_files.append(
                (split, image_path)
            )


print(f"Problematic TIFFs found: {len(problem_files)}")
print()


# ------------------------------------------------------------
# Process each TIFF
# ------------------------------------------------------------

repaired = 0
failed = 0


for index, (split, tif_path) in enumerate(
    problem_files,
    start=1
):

    print(
        f"[{index:02d}/{len(problem_files):02d}] "
        f"{tif_path.name}",
        end=" "
    )

    try:

        if not tif_path.exists():

            print("-> SOURCE NOT FOUND")
            failed += 1
            continue


        # ----------------------------------------------------
        # Open TIFF
        # ----------------------------------------------------

        with Image.open(tif_path) as img:

            # Always use FIRST frame
            img.seek(0)

            # Convert to standard RGB
            frame = img.convert("RGB")

            width, height = frame.size


            # ------------------------------------------------
            # Output JPG with SAME STEM
            # ------------------------------------------------

            output_path = (
                tif_path.parent /
                f"{tif_path.stem}.jpg"
            )


            # Avoid overwriting an existing image
            if output_path.exists():

                print(
                    f"-> JPG ALREADY EXISTS "
                    f"({output_path.name})"
                )

                # Still quarantine the bad TIFF
                quarantine_dir = (
                    QUARANTINE /
                    split
                )

                quarantine_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                shutil.move(
                    str(tif_path),
                    str(
                        quarantine_dir /
                        tif_path.name
                    )
                )

                repaired += 1
                continue


            # ------------------------------------------------
            # Save JPG
            # ------------------------------------------------

            frame.save(
                output_path,
                "JPEG",
                quality=95,
                optimize=True
            )


        # ----------------------------------------------------
        # Move original TIFF to quarantine
        # ----------------------------------------------------

        quarantine_dir = (
            QUARANTINE /
            split
        )

        quarantine_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.move(
            str(tif_path),
            str(
                quarantine_dir /
                tif_path.name
            )
        )


        repaired += 1

        print(
            f"-> REPAIRED "
            f"({width}x{height})"
        )


    except Exception as e:

        failed += 1

        print(
            f"-> FAILED: {e}"
        )


# ------------------------------------------------------------
# Remove stale YOLO caches
# ------------------------------------------------------------

print()
print("=" * 70)
print("REMOVING OLD YOLO CACHE FILES")
print("=" * 70)


cache_files = list(
    ROOT.rglob("*.cache")
)

for cache in cache_files:

    try:

        cache.unlink()

        print(
            f"Removed: {cache}"
        )

    except Exception as e:

        print(
            f"Could not remove {cache}: {e}"
        )


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("TIFF REPAIR COMPLETE")
print("=" * 70)

print(f"Problematic TIFFs : {len(problem_files)}")
print(f"Repaired          : {repaired}")
print(f"Failed            : {failed}")

print()
print(
    f"Quarantine folder:"
)
print(
    QUARANTINE
)

print("=" * 70)