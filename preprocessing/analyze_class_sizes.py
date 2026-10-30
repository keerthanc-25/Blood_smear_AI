from pathlib import Path
from collections import defaultdict
import statistics


# ============================================================
# PATH
# ============================================================

LABEL_DIR = Path(
    r"D:\Blood_Smear_AI\data\yolo_dataset_3class\labels\train"
)


CLASS_NAMES = {
    0: "RBC",
    1: "WBC",
    2: "Platelet"
}


# ============================================================
# STORAGE
# ============================================================

data = defaultdict(lambda: {
    "widths": [],
    "heights": [],
    "areas": []
})


# ============================================================
# READ LABELS
# ============================================================

for label_file in LABEL_DIR.glob("*.txt"):

    with open(label_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            if class_id not in CLASS_NAMES:
                continue

            width = float(parts[3])
            height = float(parts[4])

            area = width * height

            data[class_id]["widths"].append(width)
            data[class_id]["heights"].append(height)
            data[class_id]["areas"].append(area)


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("TRAINING OBJECT SIZE ANALYSIS")
print("=" * 70)

for class_id in [0, 1, 2]:

    name = CLASS_NAMES[class_id]

    widths = data[class_id]["widths"]
    heights = data[class_id]["heights"]
    areas = data[class_id]["areas"]

    print("\n" + "-" * 70)

    print(name)

    print("-" * 70)

    print("Objects:", len(widths))

    print(
        "Average width  :",
        round(statistics.mean(widths), 4)
    )

    print(
        "Average height :",
        round(statistics.mean(heights), 4)
    )

    print(
        "Average area   :",
        round(statistics.mean(areas), 6)
    )

    print(
        "Median width   :",
        round(statistics.median(widths), 4)
    )

    print(
        "Median height  :",
        round(statistics.median(heights), 4)
    )

    print(
        "Median area    :",
        round(statistics.median(areas), 6)
    )

    print(
        "Minimum area   :",
        round(min(areas), 6)
    )

    print(
        "Maximum area   :",
        round(max(areas), 6)
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)