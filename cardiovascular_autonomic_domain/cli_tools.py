from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from serial.tools import list_ports

from camera_rppg_features import extract_camera_rppg_features
from capture_and_analyze import (
    build_user_profile,
    capture_serial_session,
    load_dataset_from_csv,
    write_capture_csv,
    write_report_files,
)
from sequential_measurement_session import format_console_summary, load_camera_summary, print_progress, run_stepwise_analysis


DOMAIN_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOMAIN_DIR.parent
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"


def maybe_reexec_in_venv() -> None:
    if not VENV_PYTHON.exists():
        return
    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if current_python == target_python:
        return
    os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])


def list_serial_port_rows() -> list[object]:
    return list(list_ports.comports())


def describe_port(port: object) -> str:
    description = getattr(port, "description", "") or "Serial Port"
    manufacturer = getattr(port, "manufacturer", "") or ""
    extras = f" / {manufacturer}" if manufacturer else ""
    return f"{port.device} - {description}{extras}"


def choose_default_port(ports: list[object]) -> str | None:
    if not ports:
        return None
    ranked = []
    for port in ports:
        text = " ".join(
            [
                str(getattr(port, "device", "") or ""),
                str(getattr(port, "description", "") or ""),
                str(getattr(port, "manufacturer", "") or ""),
                str(getattr(port, "product", "") or ""),
            ]
        ).lower()
        score = 0
        if "arduino" in text:
            score += 10
        if "uno" in text:
            score += 8
        if "r4" in text:
            score += 6
        if "usb" in text:
            score += 2
        ranked.append((score, str(port.device)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def prompt_text(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def prompt_optional_int(prompt: str, default: str = "") -> int | None:
    raw = prompt_text(prompt, default)
    return int(raw) if raw else None


def prompt_optional_float(prompt: str, default: str = "") -> float | None:
    raw = prompt_text(prompt, default)
    return float(raw) if raw else None


def build_profile_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        age=args.age,
        sex=args.sex or "unknown",
        calibration_sbp=args.calibration_sbp,
        calibration_dbp=args.calibration_dbp,
    )


def resolve_runtime_inputs(args: argparse.Namespace) -> argparse.Namespace:
    if args.no_prompt:
        return args

    available_ports = list_serial_port_rows()
    default_port = args.port or choose_default_port(available_ports) or "COM5"

    if not args.csv_input:
        print("Available serial ports:")
        if available_ports:
            for index, port in enumerate(available_ports, start=1):
                print(f"  {index}. {describe_port(port)}")
        else:
            print("  No serial ports were detected.")
        port_input = prompt_text("Port", default_port)
        if port_input.isdigit():
            chosen_index = int(port_input) - 1
            if 0 <= chosen_index < len(available_ports):
                args.port = str(available_ports[chosen_index].device)
            else:
                args.port = default_port
        else:
            args.port = port_input

    if args.duration is None:
        args.duration = float(prompt_text("Measurement duration seconds", "60"))
    if args.age is None:
        args.age = prompt_optional_int("Age", "")
    if args.sex is None:
        args.sex = prompt_text("Sex male/female/unknown", "unknown").lower()
    if args.calibration_sbp is None:
        args.calibration_sbp = prompt_optional_float("Calibration SBP", "")
    if args.calibration_dbp is None:
        args.calibration_dbp = prompt_optional_float("Calibration DBP", "")

    return args


def run_measurement(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = (DOMAIN_DIR / "outputs" / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    capture_path: Path | None = None
    camera_summary = load_camera_summary(Path(args.camera_summary).resolve()) if args.camera_summary else None

    if args.csv_input:
        print("Using existing CSV input for sequential measurement.")
        csv_path = Path(args.csv_input).resolve()
        dataset = load_dataset_from_csv(csv_path, fallback_sample_rate_hz=args.sample_rate)
        if camera_summary is None:
            auto_camera_summary_path = csv_path.parent / "camera_rppg_summary.json"
            camera_summary = load_camera_summary(auto_camera_summary_path)
            if camera_summary is not None:
                print(f"Detected camera summary for fusion: {auto_camera_summary_path}")
    else:
        if not args.port:
            raise ValueError("A serial port is required unless --csv-input is used.")
        print("Step 0/7: capturing raw PPG data. Keep the finger stable until capture finishes.")
        samples = capture_serial_session(args.port, args.baud, float(args.duration), args.sample_rate)
        capture_path = output_dir / "capture.csv"
        write_capture_csv(capture_path, samples)
        dataset = load_dataset_from_csv(capture_path, fallback_sample_rate_hz=args.sample_rate)
        print("Signal capture complete. Starting section 1.1 to 1.7 calculations...")

    profile = build_user_profile(build_profile_namespace(args))
    report = run_stepwise_analysis(dataset, profile, camera_summary=camera_summary, progress=print_progress)
    report_path, summary_path = write_report_files(output_dir, report, capture_path=capture_path)

    print(format_console_summary(report))
    print("")
    print(f"Report JSON : {report_path}")
    print(f"Summary TXT : {summary_path}")
    if capture_path is not None:
        print(f"Capture CSV : {capture_path}")

    return report_path, summary_path


def should_pause(args: argparse.Namespace) -> bool:
    launched_without_args = len(sys.argv) == 1
    return sys.stdin.isatty() and launched_without_args and not args.no_pause


def build_measurement_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Unified cardiovascular/autonomic runner: serial capture plus section 1.1 to 1.7 sequential analysis."
    parser.add_argument("--port", help="Serial port such as COM5.")
    parser.add_argument("--baud", type=int, default=1000000, help="Serial baud rate. Default: 1000000")
    parser.add_argument("--duration", type=float, help="Measurement duration in seconds.")
    parser.add_argument("--age", type=int, help="Chronological age.")
    parser.add_argument("--sex", choices=["male", "female", "unknown"], help="Sex value for vascular age calibration.")
    parser.add_argument("--calibration-sbp", type=float, help="Optional cuff calibrated systolic baseline.")
    parser.add_argument("--calibration-dbp", type=float, help="Optional cuff calibrated diastolic baseline.")
    parser.add_argument("--sample-rate", type=float, default=200.0, help="Fallback sample rate. Default: 200")
    parser.add_argument("--csv-input", help="Optional offline CSV input for testing.")
    parser.add_argument("--camera-summary", help="Optional camera_rppg_summary.json path for camera+iPPG fusion.")
    parser.add_argument("--no-prompt", action="store_true", help="Run without asking interactive questions.")
    parser.add_argument("--no-pause", action="store_true", help="Do not wait for Enter before closing when launched without args.")


def measurement_main(argv: list[str] | None = None) -> int:
    maybe_reexec_in_venv()
    parser = argparse.ArgumentParser()
    build_measurement_parser(parser)
    args = parser.parse_args(argv)
    try:
        args = resolve_runtime_inputs(args)
        if not args.csv_input and not args.port:
            raise ValueError("No serial port was selected.")
        if args.duration is None:
            args.duration = 60.0
        if args.sex is None:
            args.sex = "unknown"
        run_measurement(args)
        return 0
    except KeyboardInterrupt:
        print("\nMeasurement cancelled by user.")
        return 1
    except Exception as exc:
        print(f"\nMeasurement failed: {exc}")
        return 1
    finally:
        if should_pause(args):
            try:
                input("\nPress Enter to close...")
            except EOFError:
                pass


def camera_extraction_main(argv: list[str] | None = None) -> int:
    maybe_reexec_in_venv()
    parser = argparse.ArgumentParser(description="Extract camera rPPG candidate features from a recorded video.")
    parser.add_argument("--video", required=True, help="Path to camera_rgb.mp4")
    parser.add_argument("--frame-csv", help="Optional path to camera_frames.csv")
    parser.add_argument("--output-dir", help="Directory for camera_rppg_features.csv and camera_rppg_summary.json")
    args = parser.parse_args(argv)

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


def main() -> int:
    maybe_reexec_in_venv()
    parser = argparse.ArgumentParser(description="Cardiovascular autonomic domain CLI tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    measurement_parser = subparsers.add_parser("measure", help="Run serial/CSV measurement and 1.1~1.7 analysis")
    build_measurement_parser(measurement_parser)

    extract_parser = subparsers.add_parser("extract-camera", help="Extract camera rPPG features from recorded video")
    extract_parser.add_argument("--video", required=True, help="Path to camera_rgb.mp4")
    extract_parser.add_argument("--frame-csv", help="Optional path to camera_frames.csv")
    extract_parser.add_argument("--output-dir", help="Directory for camera_rppg_features.csv and camera_rppg_summary.json")

    args = parser.parse_args()
    if args.command == "measure":
        return measurement_main(sys.argv[2:])
    return camera_extraction_main(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
