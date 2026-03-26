from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"


def maybe_reexec_in_venv() -> None:
    if not VENV_PYTHON.exists():
        return
    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if current_python == target_python:
        return
    os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])


maybe_reexec_in_venv()

sys.path.insert(0, str(ROOT_DIR / "cardiovascular_autonomic_domain"))

from camera_rppg_features import extract_camera_rppg_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract camera rPPG candidate features from a recorded video.")
    parser.add_argument("--video", required=True, help="Path to camera_rgb.mp4")
    parser.add_argument("--frame-csv", help="Optional path to camera_frames.csv")
    parser.add_argument("--output-dir", help="Directory for camera_rppg_features.csv and camera_rppg_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).resolve()
    frame_csv_path = Path(args.frame_csv).resolve() if args.frame_csv else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else video_path.parent

    result = extract_camera_rppg_features(video_path, output_dir, frame_timestamps_path=frame_csv_path, status_callback=print)
    print("")
    print(f"Features CSV : {result['features_csv_path']}")
    print(f"Summary JSON : {result['summary_json_path']}")
    print(f"Selected rPPG signal : {result['selected_signal']}")
    print(f"Estimated camera HR : {result['selected_hr_bpm']:.2f} bpm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
