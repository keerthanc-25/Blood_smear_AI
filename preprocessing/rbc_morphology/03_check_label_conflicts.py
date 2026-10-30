from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(
    r"D:\Blood_Smear_AI\data\rbc_morphology\raw\RBC"
)

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


records = []

for split in ["train", "val", "test"]:

    split_dir = ROOT / split

    for class_dir in split_dir.iterdir():

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for image in class_dir.rglob("*"):

            if (
                image.is_file()
                and image.suffix.lower() in EXTENSIONS
            ):

                records.append(
                    (image, class_name)
                )


print("=" * 70)
print("CHECKING DUPLICATE LABEL CONFLICTS")
print("=" * 70)

print(f"Images scanned: {len(records)}")


hash_map = defaultdict(list)

for i, (image, class_name) in enumerate(
    records, 1
):

    h = sha256(image)

    hash_map[h].append(
        (image, class_name)
    )

    if i % 2000 == 0:
        print(
            f"Hashed {i}/{len(records)}"
        )


conflicts = []

for h, items in hash_map.items():

    labels = {
        label
        for _, label in items
    }

    if len(labels) > 1:

        conflicts.append(
            (h, items)
        )


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Duplicate groups: "
    f"{sum(1 for x in hash_map.values() if len(x) > 1)}"
)

print(
    f"Cross-label duplicate groups: "
    f"{len(conflicts)}"
)


output = (
    Path(
        r"D:\Blood_Smear_AI\data\rbc_morphology\metadata"
    )
    / "label_conflicts.txt"
)

with open(
    output,
    "w",
    encoding="utf-8"
) as f:

    for index, (h, items) in enumerate(
        conflicts,
        1
    ):

        f.write(
            f"\nGROUP {index}\n"
        )

        f.write(
            f"SHA256: {h}\n"
        )

        for image, label in items:

            f.write(
                f"{label} | {image}\n"
            )


print(
    f"\nReport saved to:\n{output}"
)

print("\nOriginal dataset was NOT modified.")