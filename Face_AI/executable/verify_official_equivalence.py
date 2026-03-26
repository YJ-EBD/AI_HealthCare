import csv
import io
import json
import zipfile
from pathlib import Path


def load_manifest(face_ai_root):
    manifest_path = face_ai_root / "official_expected_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def find_reference_dataset(reference_root):
    children = [path for path in reference_root.iterdir() if path.is_dir()]
    if len(children) != 1:
        raise RuntimeError(
            f"Expected exactly one dataset directory under {reference_root}, found {len(children)}"
        )
    return children[0]


def count_jpg_subjects(zip_path):
    image_count = 0
    subjects = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            image_count += 1
            parts = name.split("/")
            if len(parts) >= 3:
                subjects.add(parts[1])
    return image_count, len(subjects)


def count_json(zip_path):
    json_count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".json"):
                json_count += 1
    return json_count


def count_csv_cells(zip_path):
    measurement_total = None
    metadata_total = None
    measurement_rows = None
    metadata_rows = None

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue

            with zf.open(name) as fh:
                rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8-sig")))

            field_count = len(rows[0].keys()) if rows else 0

            low_name = name.lower()
            if low_name.endswith("measurement_data.csv"):
                measurement_rows = len(rows)
                measurement_total = len(rows) * max(field_count - 1, 0)
            elif low_name.endswith("meta_data.csv"):
                metadata_rows = len(rows)
                metadata_total = len(rows) * field_count

    return measurement_rows, measurement_total, metadata_rows, metadata_total


def compare(actual, expected):
    return {
        "actual": actual,
        "expected": expected,
        "match": actual == expected,
        "delta": actual - expected,
    }


def main():
    face_ai_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(face_ai_root)
    reference_root = face_ai_root / "reference"
    dataset_root = find_reference_dataset(reference_root)

    archives = {path.name: path for path in dataset_root.rglob("*") if path.is_file()}

    ts_images, ts_subjects = count_jpg_subjects(archives["TS.zip"])
    vs_images, vs_subjects = count_jpg_subjects(archives["VS.zip"])
    tl_json = count_json(archives["TL.zip"])
    vl_json = count_json(archives["VL.zip"])
    measurement_rows, measurement_total, metadata_rows, metadata_total = count_csv_cells(
        archives["Other.zip"]
    )

    actual = {
        "face_images_total": ts_images + vs_images,
        "label_records_total": tl_json + vl_json,
        "measurement_records_total": measurement_total,
        "metadata_records_total": metadata_total,
        "subjects_total_local_images": ts_subjects + vs_subjects,
        "subjects_total_inferred_from_metadata": measurement_rows,
    }

    expected = manifest["expected_counts"]
    report = {
        "dataset_name": manifest["dataset_name"],
        "source": manifest["source"],
        "actual": actual,
        "comparisons": {
            "face_images_total": compare(
                actual["face_images_total"], expected["face_images_total"]
            ),
            "label_records_total": compare(
                actual["label_records_total"], expected["label_records_total"]
            ),
            "measurement_records_total": compare(
                actual["measurement_records_total"], expected["measurement_records_total"]
            ),
            "metadata_records_total": compare(
                actual["metadata_records_total"], expected["metadata_records_total"]
            ),
            "subjects_total_inferred": compare(
                actual["subjects_total_inferred_from_metadata"],
                expected["subjects_total_inferred"],
            ),
        },
        "derived_gap_hint": {
            "missing_images_if_official_counts_are_correct": expected["face_images_total"]
            - actual["face_images_total"],
            "missing_labels_if_official_counts_are_correct": expected["label_records_total"]
            - actual["label_records_total"],
            "missing_subjects_in_images_vs_metadata": actual["subjects_total_inferred_from_metadata"]
            - actual["subjects_total_local_images"],
        },
        "all_core_counts_match": all(
            item["match"] for item in [
                compare(actual["face_images_total"], expected["face_images_total"]),
                compare(actual["label_records_total"], expected["label_records_total"]),
                compare(actual["measurement_records_total"], expected["measurement_records_total"]),
                compare(actual["metadata_records_total"], expected["metadata_records_total"]),
            ]
        ),
        "notes": manifest["notes"],
    }

    output_path = face_ai_root / "executable" / "output" / "official_equivalence_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
