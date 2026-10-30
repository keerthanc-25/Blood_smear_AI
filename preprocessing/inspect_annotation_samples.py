import os
import csv

PROJECT_DIR = r"D:\Blood_Smear_AI"

ANALYSIS_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "annotation_analysis.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "annotation_samples.txt"
)


def main():

    print("=" * 70)
    print("STEP 2B - INSPECT ANNOTATION SAMPLES")
    print("=" * 70)

    if not os.path.exists(ANALYSIS_FILE):
        print("\nERROR:")
        print(ANALYSIS_FILE)
        print("\nRun analyze_annotations.py first.")
        return

    with open(
        ANALYSIS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(csv.DictReader(f))

    selected = []

    # Get examples of each important format
    wanted = {
        "OTHER_TXT": 10,
        "OTHER_XML": 10,
        "PASCAL_VOC_XML": 5,
        "YOLO_TXT": 5
    }

    counts = {
        key: 0
        for key in wanted
    }

    for row in rows:

        fmt = row["format_guess"]

        if fmt not in wanted:
            continue

        if counts[fmt] >= wanted[fmt]:
            continue

        selected.append(row)
        counts[fmt] += 1

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as out:

        out.write(
            "ANNOTATION SAMPLE INSPECTION\n"
        )

        out.write(
            "=" * 70 + "\n\n"
        )

        for row in selected:

            out.write(
                f"FORMAT: {row['format_guess']}\n"
            )

            out.write(
                f"FILE: {row['annotation_file']}\n"
            )

            out.write(
                f"EXTENSION: {row['extension']}\n"
            )

            out.write(
                "-" * 70 + "\n"
            )

            # Read the actual file
            full_path = os.path.join(
                PROJECT_DIR,
                row["annotation_file"]
            )

            try:

                with open(
                    full_path,
                    "r",
                    encoding="utf-8",
                    errors="replace"
                ) as f:

                    content = f.read(3000)

                out.write(content)

            except Exception as e:

                out.write(
                    f"ERROR READING FILE: {e}\n"
                )

            out.write(
                "\n\n"
                + "=" * 70
                + "\n\n"
            )

    print("\nSamples collected:")

    for fmt, count in counts.items():

        print(
            f"{fmt:20s}: {count}"
        )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nSTEP 2B COMPLETE")


if __name__ == "__main__":
    main()