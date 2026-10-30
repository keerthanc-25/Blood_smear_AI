import os
import csv
from collections import Counter

PROJECT_DIR = r"D:\Blood_Smear_AI"

INVENTORY_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "dataset_inventory.csv"
)

REPORT_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "annotation_analysis.csv"
)


def analyze_txt(path):
    """
    Inspect TXT annotations without assuming they are YOLO.
    """

    result = {
        "format_guess": "TXT",
        "lines": 0,
        "valid_yolo_like": 0,
        "invalid_lines": 0,
        "sample": ""
    }

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        result["lines"] = len(lines)

        samples = []

        for line in lines[:5]:

            parts = line.split()

            if len(parts) == 5:

                try:
                    values = [
                        float(x)
                        for x in parts
                    ]

                    class_id = int(float(parts[0]))

                    if (
                        class_id >= 0
                        and 0 <= values[1] <= 1
                        and 0 <= values[2] <= 1
                        and 0 < values[3] <= 1
                        and 0 < values[4] <= 1
                    ):
                        result["valid_yolo_like"] += 1

                except ValueError:
                    pass

            samples.append(line)

        result["invalid_lines"] = (
            len(lines)
            - result["valid_yolo_like"]
        )

        if samples:
            result["sample"] = " | ".join(
                samples[:3]
            )

        if (
            result["lines"] > 0
            and result["valid_yolo_like"] == result["lines"]
        ):
            result["format_guess"] = "YOLO_TXT"

        elif result["lines"] > 0:
            result["format_guess"] = "OTHER_TXT"

    except Exception as e:

        result["format_guess"] = (
            "ERROR: " + str(e)
        )

    return result


def analyze_xml(path):

    result = {
        "format_guess": "XML",
        "lines": 0,
        "valid_yolo_like": 0,
        "invalid_lines": 0,
        "sample": ""
    }

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            content = f.read(10000)

        result["lines"] = content.count("\n")

        lower = content.lower()

        if (
            "<annotation" in lower
            and "<object" in lower
            and "<bndbox" in lower
        ):
            result["format_guess"] = "PASCAL_VOC_XML"

        elif "<annotation" in lower:
            result["format_guess"] = "XML_ANNOTATION"

        else:
            result["format_guess"] = "OTHER_XML"

        result["sample"] = (
            content[:500]
            .replace("\n", " ")
        )

    except Exception as e:

        result["format_guess"] = (
            "ERROR: " + str(e)
        )

    return result


def analyze_json(path):

    result = {
        "format_guess": "JSON",
        "lines": 0,
        "valid_yolo_like": 0,
        "invalid_lines": 0,
        "sample": ""
    }

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            content = f.read()

        lower = content.lower()

        if (
            '"images"' in lower
            and '"annotations"' in lower
            and '"categories"' in lower
        ):
            result["format_guess"] = "COCO_JSON"

        elif (
            '"annotations"' in lower
        ):
            result["format_guess"] = "JSON_ANNOTATION"

        else:
            result["format_guess"] = "OTHER_JSON"

        result["lines"] = content.count("\n")

        result["sample"] = (
            content[:500]
            .replace("\n", " ")
        )

    except Exception as e:

        result["format_guess"] = (
            "ERROR: " + str(e)
        )

    return result


def analyze_generic(path):

    extension = os.path.splitext(
        path
    )[1].lower()

    return {
        "format_guess": extension.upper(),
        "lines": 0,
        "valid_yolo_like": 0,
        "invalid_lines": 0,
        "sample": ""
    }


def main():

    print("=" * 75)
    print("STEP 2A - ANNOTATION FORMAT ANALYSIS")
    print("=" * 75)

    if not os.path.exists(INVENTORY_FILE):

        print("\nERROR:")
        print("Dataset inventory not found:")

        print(INVENTORY_FILE)

        print(
            "\nRun Step 1 first."
        )

        return

    with open(
        INVENTORY_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    print(
        "\nImages in inventory:",
        f"{len(rows):,}"
    )

    # --------------------------------------------------------
    # Get unique annotation files
    # --------------------------------------------------------

    annotation_files = set()

    for row in rows:

        value = row.get(
            "annotation_files",
            ""
        )

        if not value:
            continue

        for item in value.split(";"):

            item = item.strip()

            if item:
                annotation_files.add(item)

    annotation_files = sorted(
        annotation_files
    )

    print(
        "Unique annotation files:",
        f"{len(annotation_files):,}"
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    reports = []
    format_counts = Counter()

    print("\nAnalyzing annotation files...")
    print("Progress shown every 10,000 files.\n")

    for index, relative_path in enumerate(
        annotation_files,
        1
    ):

        full_path = os.path.join(
            PROJECT_DIR,
            relative_path
        )

        extension = os.path.splitext(
            full_path
        )[1].lower()

        if extension == ".txt":

            result = analyze_txt(
                full_path
            )

        elif extension == ".xml":

            result = analyze_xml(
                full_path
            )

        elif extension == ".json":

            result = analyze_json(
                full_path
            )

        else:

            result = analyze_generic(
                full_path
            )

        format_guess = result[
            "format_guess"
        ]

        format_counts[
            format_guess
        ] += 1

        reports.append({

            "annotation_file":
                relative_path,

            "extension":
                extension,

            "format_guess":
                format_guess,

            "lines":
                result["lines"],

            "valid_yolo_like":
                result["valid_yolo_like"],

            "invalid_lines":
                result["invalid_lines"],

            "sample":
                result["sample"]
        })

        if (
            index % 10000 == 0
            or index == len(annotation_files)
        ):

            print(
                f"Analyzed "
                f"{index:,}/{len(annotation_files):,}"
            )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT_FILE),
        exist_ok=True
    )

    fieldnames = [
        "annotation_file",
        "extension",
        "format_guess",
        "lines",
        "valid_yolo_like",
        "invalid_lines",
        "sample"
    ]

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(reports)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("ANNOTATION FORMAT SUMMARY")
    print("=" * 75)

    for format_name, count in (
        format_counts.most_common()
    ):

        print(
            f"{format_name:30s} : {count:,}"
        )

    print("\n" + "=" * 75)
    print("OUTPUT")
    print("=" * 75)

    print("\nAnalysis report:")
    print(REPORT_FILE)

    print("\n" + "=" * 75)
    print("STEP 2A COMPLETE")
    print("=" * 75)

    print(
        "\nIMPORTANT:"
    )

    print(
        "No annotation files were modified."
    )

    print(
        "No annotations were converted."
    )

    print(
        "No labels were created."
    )

    print(
        "\nUse the format summary to decide the next conversion step."
    )


if __name__ == "__main__":
    main()