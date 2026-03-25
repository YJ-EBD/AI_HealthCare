from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import serial

from cardiovascular_metrics import SignalDataset, UserProfile, analyze_dataset, estimate_sample_rate


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    number = parse_float(value)
    return None if number is None else int(number)


def parse_arduino_line(line: str, fallback_sample_rate_hz: float, implicit_index: int) -> dict[str, float | int] | None:
    line = line.strip()
    if not line:
        return None

    if line.startswith("type,") or line.startswith("INFO,"):
        return None

    if line.startswith("RAW,") or line.startswith("STAT,"):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            return None
        timestamp_ms = parse_float(parts[1])
        sample_index = parse_int(parts[2])
        ppg_raw = parse_float(parts[3])
        beat_raw = parse_float(parts[4])
        ppg_v = parse_float(parts[5])
        beat_v = parse_float(parts[6])
        return {
            "timestamp_s": (timestamp_ms or 0.0) / 1000.0,
            "sample": sample_index if sample_index is not None else implicit_index,
            "ppg": ppg_raw if ppg_raw is not None else (ppg_v or 0.0),
            "beat": beat_raw if beat_raw is not None else (beat_v or 0.0),
            "ppg_raw": ppg_raw if ppg_raw is not None else math.nan,
            "beat_raw": beat_raw if beat_raw is not None else math.nan,
            "ppg_v": ppg_v if ppg_v is not None else math.nan,
            "beat_v": beat_v if beat_v is not None else math.nan,
        }

    if line.startswith("BEAT,"):
        return None

    parts = [part.strip() for part in line.split(",")]
    if len(parts) == 2:
        ppg_v = parse_float(parts[0])
        beat_v = parse_float(parts[1])
        if ppg_v is None or beat_v is None:
            return None
        return {
            "timestamp_s": implicit_index / fallback_sample_rate_hz,
            "sample": implicit_index,
            "ppg": ppg_v,
            "beat": beat_v,
            "ppg_raw": math.nan,
            "beat_raw": math.nan,
            "ppg_v": ppg_v,
            "beat_v": beat_v,
        }

    return None


def capture_serial_session(port: str, baud: int, duration_s: float, fallback_sample_rate_hz: float) -> list[dict[str, float | int]]:
    samples: list[dict[str, float | int]] = []
    started_at = time.time()

    with serial.Serial(port=port, baudrate=baud, timeout=0.3) as connection:
        connection.reset_input_buffer()
        time.sleep(1.0)
        print(f"Capturing serial data from {port} at {baud} baud for {duration_s:.1f} seconds...")

        while time.time() - started_at < duration_s:
            raw_line = connection.readline()
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore")
            parsed = parse_arduino_line(line, fallback_sample_rate_hz=fallback_sample_rate_hz, implicit_index=len(samples))
            if parsed is not None:
                samples.append(parsed)
                if len(samples) % 500 == 0:
                    elapsed = time.time() - started_at
                    print(f"  collected {len(samples)} samples in {elapsed:.1f}s")

    if not samples:
        raise RuntimeError("No usable samples were captured from the serial port.")

    return samples


def write_capture_csv(path: Path, samples: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp_s", "sample", "ppg", "beat", "ppg_raw", "beat_raw", "ppg_v", "beat_v"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)


def load_dataset_from_csv(path: Path, fallback_sample_rate_hz: float) -> SignalDataset:
    timestamps_s: list[float] = []
    ppg_values: list[float] = []
    beat_values: list[float] = []
    aux_values: list[float] = []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows were found in {path}.")

    for index, row in enumerate(rows):
        timestamp_s = parse_float(row.get("timestamp_s"))
        if timestamp_s is None:
            sample_value = parse_int(row.get("sample"))
            timestamp_s = (sample_value if sample_value is not None else index) / fallback_sample_rate_hz

        ppg_value = parse_float(row.get("ppg_raw"))
        if ppg_value is None or math.isnan(ppg_value):
            ppg_value = parse_float(row.get("ppg"))
        if ppg_value is None:
            ppg_value = parse_float(row.get("ppg_v"))
        if ppg_value is None:
            raise ValueError("Unable to find a usable PPG column in the CSV input.")

        beat_value = parse_float(row.get("beat_raw"))
        if beat_value is None or math.isnan(beat_value):
            beat_value = parse_float(row.get("beat"))
        if beat_value is None:
            beat_value = parse_float(row.get("beat_v"))
        if beat_value is None:
            beat_value = 0.0

        aux_value = parse_float(row.get("aux"))
        if aux_value is None:
            aux_value = parse_float(row.get("ppg_aux_raw"))

        timestamps_s.append(timestamp_s)
        ppg_values.append(ppg_value)
        beat_values.append(beat_value)
        if aux_value is not None:
            aux_values.append(aux_value)

    timestamp_array = np.asarray(timestamps_s, dtype=float)
    ppg_array = np.asarray(ppg_values, dtype=float)
    beat_array = np.asarray(beat_values, dtype=float)
    aux_array = np.asarray(aux_values, dtype=float) if len(aux_values) == len(ppg_values) else None

    sample_rate_hz = estimate_sample_rate(timestamp_array)
    if sample_rate_hz <= 0.0:
        sample_rate_hz = fallback_sample_rate_hz

    return SignalDataset(
        timestamps_s=timestamp_array,
        ppg=ppg_array,
        beat=beat_array,
        aux=aux_array,
        sample_rate_hz=sample_rate_hz,
    )


def write_report_files(output_dir: Path, report: dict[str, Any], capture_path: Path | None = None) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis_report.json"
    summary_path = output_dir / "summary.txt"

    report_payload = dict(report)
    report_payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    if capture_path is not None:
        report_payload["capture_csv"] = str(capture_path)

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

    heart_rate = report["heart_rate"]
    hrv = report["hrv"]
    stress = report["stress"]
    circulation = report["circulation"]
    vascular_health = report["vascular_health"]
    vascular_age = report["vascular_age"]
    blood_pressure = report["blood_pressure"]

    lines = [
        "Cardiovascular and autonomic analysis summary",
        "",
        f"1.1 Heart rate: {heart_rate['heart_rate_bpm']:.2f} bpm",
        f"1.2 HRV: SDNN {hrv['sdnn_ms']:.2f} ms | RMSSD {hrv['rmssd_ms']:.2f} ms | pNN50 {hrv['pnn50']:.2f}% | score {hrv['hrv_score']:.2f}",
        f"1.3 Stress: score {stress['stress_score']:.2f} / 100 | state {stress['stress_state']}",
        f"1.4 Circulation: score {circulation['circulation_score']:.2f} | rise time {circulation['median_rise_time_s']:.3f}s",
        f"1.5 Vascular health: score {vascular_health['vascular_health_score']:.2f} | reflection index {vascular_health['reflection_index']}",
        f"1.6 Vascular age: estimate {vascular_age['vascular_age_estimate']:.1f} years | gap {vascular_age['vascular_age_gap']:+.1f}",
        f"1.7 Blood pressure: {blood_pressure['estimated_sbp']:.1f}/{blood_pressure['estimated_dbp']:.1f} mmHg | trend {blood_pressure['blood_pressure_trend']}",
        "",
        "Warnings:",
    ]
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None")

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return report_path, summary_path


def build_user_profile(args: argparse.Namespace) -> UserProfile:
    return UserProfile(
        age=args.age,
        sex=args.sex,
        calibration_sbp=args.calibration_sbp,
        calibration_dbp=args.calibration_dbp,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture PSL-iPPG2C data and calculate the cardiovascular/autonomic metrics from section 1.1 to 1.7."
    )
    parser.add_argument("--port", help="Serial port such as COM3. When omitted, --csv-input must be used.")
    parser.add_argument("--baud", type=int, default=1000000, help="Serial baud rate. Default: 1000000")
    parser.add_argument("--duration", type=float, default=60.0, help="Capture duration in seconds. Default: 60")
    parser.add_argument("--csv-input", help="Analyze an existing capture CSV instead of reading from serial.")
    parser.add_argument("--sample-rate", type=float, default=200.0, help="Fallback sample rate. Default: 200 Hz")
    parser.add_argument("--age", type=int, help="Chronological age used for vascular age estimation.")
    parser.add_argument("--sex", default="unknown", help="male, female, or unknown")
    parser.add_argument("--calibration-sbp", type=float, help="Personal cuff-calibrated systolic baseline.")
    parser.add_argument("--calibration-dbp", type=float, help="Personal cuff-calibrated diastolic baseline.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Directory for captured CSV and report files.",
    )
    args = parser.parse_args()

    if not args.port and not args.csv_input:
        parser.error("Provide either --port for live capture or --csv-input for offline analysis.")

    output_dir = Path(args.output_dir).resolve()
    capture_path: Path | None = None

    if args.csv_input:
        dataset = load_dataset_from_csv(Path(args.csv_input).resolve(), fallback_sample_rate_hz=args.sample_rate)
    else:
        samples = capture_serial_session(args.port, args.baud, args.duration, args.sample_rate)
        capture_path = output_dir / "capture.csv"
        write_capture_csv(capture_path, samples)
        dataset = load_dataset_from_csv(capture_path, fallback_sample_rate_hz=args.sample_rate)

    report = analyze_dataset(dataset, build_user_profile(args))
    report_path, summary_path = write_report_files(output_dir, report, capture_path=capture_path)

    print("")
    print("Analysis complete.")
    print(f"Report JSON : {report_path}")
    print(f"Summary TXT : {summary_path}")
    if capture_path is not None:
        print(f"Capture CSV : {capture_path}")


if __name__ == "__main__":
    main()
