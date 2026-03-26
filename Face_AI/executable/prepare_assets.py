import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


CHECKPOINT_FILES = [
    "2.AI학습모델파일\\checkpoint\\class\\100%\\1,2,3\\Evaluation.txt",
    "2.AI학습모델파일\\checkpoint\\class\\100%\\1,2,3\\1\\state_dict.bin",
    "2.AI학습모델파일\\checkpoint\\class\\100%\\1,2,3\\2\\state_dict.bin",
    "2.AI학습모델파일\\checkpoint\\class\\100%\\1,2,3\\3\\state_dict.bin",
    "2.AI학습모델파일\\checkpoint\\class\\100%\\1,2,3\\5\\state_dict.bin",
    "2.AI학습모델파일\\checkpoint\\class\\100%\\1,2,3\\7\\state_dict.bin",
    "2.AI학습모델파일\\checkpoint\\class\\100%\\1,2,3\\8\\state_dict.bin",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-training", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def extract_zip(archive_path, target_dir, force):
    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        print(f"skip zip {archive_path.name} -> {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(target_dir)
    print(f"extracted zip {archive_path.name} -> {target_dir}")


def extract_checkpoints(egg_path, target_root, temp_root, force):
    existing = target_root / "1" / "state_dict.bin"
    if existing.exists() and not force:
        print(f"skip checkpoints -> {target_root}")
        return

    bz = Path(r"C:\Program Files\Bandizip\bz.exe")
    if not bz.exists():
        raise FileNotFoundError(
            "Bandizip CLI not found at C:\\Program Files\\Bandizip\\bz.exe"
        )

    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    for archive_member in CHECKPOINT_FILES:
        subprocess.run(
            [str(bz), "x", str(egg_path), archive_member, "-aoa", "-y"],
            cwd=temp_root,
            check=True,
        )

    extracted_root = temp_root / "2.AI학습모델파일" / "checkpoint" / "class" / "100%" / "1,2,3"
    if not extracted_root.exists():
        raise FileNotFoundError(f"Expected extracted root missing: {extracted_root}")

    if target_root.exists() and force:
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    for item in extracted_root.iterdir():
        destination = target_root / item.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(item), str(destination))

    print(f"extracted checkpoints -> {target_root}")


def main():
    args = parse_args()
    face_ai_root = Path(__file__).resolve().parents[1]
    reference_root = face_ai_root / "reference"
    dataset_root = next(reference_root.iterdir())

    archives = {path.name: path for path in dataset_root.rglob("*") if path.is_file()}
    data_root = face_ai_root / "data"

    extract_zip(archives["VS.zip"], data_root / "validation" / "images", args.force)
    extract_zip(archives["VL.zip"], data_root / "validation" / "labels", args.force)
    extract_zip(archives["Other.zip"], data_root / "metadata", args.force)

    if args.include_training:
        extract_zip(archives["TS.zip"], data_root / "training" / "images", args.force)
        extract_zip(archives["TL.zip"], data_root / "training" / "labels", args.force)

    extract_checkpoints(
        archives["1.모델.egg"],
        face_ai_root / "model" / "checkpoint" / "class" / "released" / "1_2_3",
        face_ai_root / "model" / "_tmp_extract",
        args.force,
    )


if __name__ == "__main__":
    main()
