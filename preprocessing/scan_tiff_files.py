from pathlib import Path
from PIL import Image

ROOT = Path(r"D:\Blood_Smear_AI\data\final_dataset")

bad = []
multi_page = 0
normal = 0

print("=" * 70)
print("TIFF DATASET CHECK")
print("=" * 70)

files = []

for split in ["train", "val", "test"]:
    folder = ROOT / "images" / split

    files.extend(
        [
            (split, p)
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".tif", ".tiff"}
        ]
    )

print(f"TIFF files found: {len(files)}")
print()

for index, (split, path) in enumerate(files, 1):

    try:
        with Image.open(path) as img:

            frames = getattr(img, "n_frames", 1)

            sizes = []

            for frame in range(frames):
                img.seek(frame)
                sizes.append(img.size)

            unique_sizes = set(sizes)

            if frames > 1:
                multi_page += 1

                print(
                    f"[MULTI-PAGE] {index}/{len(files)} "
                    f"{split}/{path.name} "
                    f"frames={frames} "
                    f"sizes={unique_sizes}"
                )

            if len(unique_sizes) > 1:
                bad.append(
                    (
                        split,
                        path,
                        frames,
                        unique_sizes
                    )
                )

            else:
                normal += 1

    except Exception as e:

        bad.append(
            (
                split,
                path,
                "READ_ERROR",
                str(e)
            )
        )

        print(
            f"[READ ERROR] {split}/{path.name}: {e}"
        )


print()
print("=" * 70)
print("SCAN COMPLETE")
print("=" * 70)

print(f"Total TIFF/TIF       : {len(files)}")
print(f"Multi-page TIFFs     : {multi_page}")
print(f"Normal TIFFs         : {normal}")
print(f"Problematic files    : {len(bad)}")

print()

if bad:

    report = ROOT / "tiff_problem_files.txt"

    with open(report, "w", encoding="utf-8") as f:

        for item in bad:
            f.write(
                " | ".join(
                    str(x) for x in item
                )
                + "\n"
            )

    print("Problem report:")
    print(report)

    print()
    print("PROBLEM FILES:")

    for item in bad:
        print(item)

else:

    print("No TIFF files with inconsistent frame dimensions found.")

print("=" * 70)