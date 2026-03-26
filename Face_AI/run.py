from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FACE_AI_ROOT = Path(__file__).resolve().parent
EXECUTABLE_DIR = FACE_AI_ROOT / "executable"
PREPARE_SCRIPT = EXECUTABLE_DIR / "prepare_assets.py"
DATASET_EVAL_SCRIPT = EXECUTABLE_DIR / "run_validation_classification.py"
LIVE_UI_SCRIPT = FACE_AI_ROOT / "live_ui.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Face_AI entry point. By default this prepares assets and launches "
            "the PySide6 live webcam UI."
        )
    )
    parser.add_argument(
        "--mode",
        default="live",
        choices=["live", "dataset-eval"],
        help="Choose between the live webcam UI and the validation dataset evaluator.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip prepare_assets.py if data/checkpoints are already ready.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Rebuild prepared assets even if they already exist.",
    )
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="Also expand the training images/labels during prepare.",
    )
    parser.add_argument("--camera-index", default=0, type=int)
    parser.add_argument("--window-width", default=900, type=int)
    parser.add_argument("--window-height", default=1440, type=int)
    parser.add_argument("--disable-reference-calibration", action="store_true")
    return parser.parse_known_args()


def run_step(command: list[str], title: str) -> None:
    print(f"[Face_AI] {title}")
    print(f"[Face_AI] command: {' '.join(command)}")
    completed = subprocess.run(command, cwd=FACE_AI_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args, passthrough = parse_args()

    if not args.skip_prepare:
        prepare_cmd = [sys.executable, str(PREPARE_SCRIPT)]
        if args.include_training:
            prepare_cmd.append("--include-training")
        if args.force_prepare:
            prepare_cmd.append("--force")
        run_step(prepare_cmd, "자산 준비를 시작합니다.")

    if args.mode == "dataset-eval":
        run_cmd = [sys.executable, str(DATASET_EVAL_SCRIPT), *passthrough]
        run_step(run_cmd, "검증 데이터셋 분류 평가를 시작합니다.")
        return 0

    run_cmd = [
        sys.executable,
        str(LIVE_UI_SCRIPT),
        "--camera-index",
        str(args.camera_index),
        "--window-width",
        str(args.window_width),
        "--window-height",
        str(args.window_height),
        *passthrough,
    ]
    if args.disable_reference_calibration:
        run_cmd.append("--disable-reference-calibration")
    run_step(run_cmd, "PySide6 실시간 UI를 시작합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
