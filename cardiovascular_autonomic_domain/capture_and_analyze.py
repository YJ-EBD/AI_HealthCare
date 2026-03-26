from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import serial

from cardiovascular_metrics import SignalDataset, UserProfile, analyze_dataset, estimate_sample_rate
from diagnostics import log_event, log_exception


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


def capture_serial_session(
    port: str,
    baud: int,
    duration_s: float,
    fallback_sample_rate_hz: float,
    status_callback: Callable[[str], None] | None = None,
    retry_count: int = 2,
    reopen_delay_s: float = 1.0,
    no_data_timeout_s: float = 5.0,
) -> list[dict[str, float | int]]:
    def emit_status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)
        else:
            print(message)
    last_error: BaseException | None = None

    for attempt in range(retry_count + 1):
        samples: list[dict[str, float | int]] = []
        started_at = time.time()
        last_data_at = started_at
        attempt_label = f"{attempt + 1}/{retry_count + 1}"
        try:
            log_event(
                "serial_capture",
                "시리얼 수집을 시작합니다.",
                details={"port": port, "baud": baud, "attempt": attempt + 1, "duration_s": duration_s},
            )
            with serial.Serial(port=port, baudrate=baud, timeout=0.3) as connection:
                connection.reset_input_buffer()
                time.sleep(1.0)
                emit_status(f"{port} 포트에서 {baud} baud로 {duration_s:.1f}초 동안 시리얼 데이터를 수집합니다... (시도 {attempt_label})")

                while time.time() - started_at < duration_s:
                    raw_line = connection.readline()
                    now = time.time()
                    if not raw_line:
                        if now - last_data_at >= no_data_timeout_s:
                            raise TimeoutError(f"{no_data_timeout_s:.1f}초 동안 시리얼 데이터가 들어오지 않았습니다.")
                        continue
                    line = raw_line.decode("utf-8", errors="ignore")
                    parsed = parse_arduino_line(line, fallback_sample_rate_hz=fallback_sample_rate_hz, implicit_index=len(samples))
                    if parsed is not None:
                        last_data_at = now
                        samples.append(parsed)
                        if len(samples) % 500 == 0:
                            elapsed = now - started_at
                            emit_status(f"  {elapsed:.1f}초 동안 {len(samples)}개 샘플을 수집했습니다")
            if samples:
                log_event(
                    "serial_capture",
                    "시리얼 수집이 완료되었습니다.",
                    details={"port": port, "sample_count": len(samples), "attempt": attempt + 1},
                )
                return samples
            raise RuntimeError("시리얼 포트에서 사용할 수 있는 샘플을 하나도 수집하지 못했습니다.")
        except BaseException as exc:  # noqa: BLE001
            last_error = exc
            log_exception(
                "serial_capture",
                exc,
                details={"port": port, "baud": baud, "attempt": attempt + 1, "retry_count": retry_count},
            )
            if attempt >= retry_count:
                break
            emit_status(f"시리얼 연결을 다시 시도합니다. ({attempt_label} 실패: {exc})")
            time.sleep(reopen_delay_s)

    raise RuntimeError(f"시리얼 수집에 실패했습니다: {last_error}") from last_error


def write_capture_csv(path: Path, samples: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    standard_fieldnames = ["timestamp_s", "sample", "ppg", "beat", "ppg_raw", "beat_raw", "ppg_v", "beat_v"]
    extra_fieldnames: list[str] = []
    for sample in samples:
        for key in sample:
            if key not in standard_fieldnames and key not in extra_fieldnames:
                extra_fieldnames.append(key)
    fieldnames = [*standard_fieldnames, *extra_fieldnames]
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
        raise ValueError(f"{path} 파일에서 읽을 수 있는 행이 없습니다.")

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
            raise ValueError("CSV 입력에서 사용할 수 있는 PPG 컬럼을 찾지 못했습니다.")

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
    metadata = report.get("metadata") or {}
    camera = report.get("camera") or {}
    ml_report = report.get("ml") or {}
    quality = report.get("quality") or {}
    no_read_outputs = quality.get("no_read_outputs") or []
    quality_outputs = quality.get("outputs") or {}
    quality_line = " | ".join(
        f"{key} {float(value.get('confidence_score') or 0.0):.0f}"
        for key, value in quality_outputs.items()
    )

    lines = [
        "심혈관 및 자율신경 분석 요약",
        "",
        f"측정 모드: {metadata.get('measurement_mode_label', 'iPPG 단독')}",
        f"ML 추론: {'적용' if ml_report.get('available') else '미적용'}",
        f"전체 신뢰도: {float(quality.get('overall_confidence_score') or 0.0):.2f} / 100",
        f"1.1 심박수: {heart_rate['heart_rate_bpm']:.2f} bpm",
        f"1.2 HRV: SDNN {hrv['sdnn_ms']:.2f} ms | RMSSD {hrv['rmssd_ms']:.2f} ms | pNN50 {hrv['pnn50']:.2f}% | 점수 {hrv['hrv_score']:.2f}",
        f"1.3 스트레스: 점수 {stress['stress_score']:.2f} / 100 | 상태 {stress['stress_state']}",
        f"1.4 순환: 점수 {circulation['circulation_score']:.2f} | 상승 시간 {circulation['median_rise_time_s']:.3f}s",
        f"1.5 혈관 건강: 점수 {vascular_health['vascular_health_score']:.2f} | 반사 지수 {vascular_health['reflection_index']}",
        f"1.6 혈관 나이: 추정 {vascular_age['vascular_age_estimate']:.1f}세 | 차이 {vascular_age['vascular_age_gap']:+.1f}",
        f"1.7 혈압: {blood_pressure['estimated_sbp']:.1f}/{blood_pressure['estimated_dbp']:.1f} mmHg | 추세 {blood_pressure['blood_pressure_trend']}",
        f"출력 신뢰도: {quality_line or '-'}",
        f"무응답 권장: {', '.join(no_read_outputs) if no_read_outputs else '없음'}",
        "",
        "경고:",
    ]
    if camera.get("available"):
        camera_hr_text = f"{float(camera['camera_hr_bpm']):.2f} bpm" if camera.get("camera_hr_bpm") is not None else "미검출"
        face_text = f"{float(camera['face_detection_ratio']) * 100.0:.1f}%" if camera.get("face_detection_ratio") is not None else "-"
        signal_text = str(camera.get("selected_signal_label") or "-")
        perfusion_text = f"{float(camera.get('camera_perfusion_proxy_score') or 0.0):.1f}"
        vascular_text = f"{float(camera.get('camera_vascular_proxy_score') or 0.0):.1f}"
        lines.insert(
            2,
            f"카메라 보조: {camera.get('measurement_mode_label', '카메라 보조 분석')} | 카메라 HR {camera_hr_text} | 얼굴 검출률 {face_text} | 신호 {signal_text} | 관류 프록시 {perfusion_text} | 혈관 프록시 {vascular_text}",
        )
    if ml_report.get("available"):
        lines.insert(
            3 if camera.get("available") else 2,
            f"ML 모델: {ml_report.get('bundle_version') or '-'} | {'기본 번들' if ml_report.get('bootstrap_bundle') else '사용자 학습 번들'}",
        )
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- 없음")

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
        parser.error("--port로 라이브 측정을 하거나 --csv-input으로 오프라인 분석 파일을 지정하세요.")

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
    print("분석이 완료되었습니다.")
    print(f"리포트 JSON : {report_path}")
    print(f"요약 TXT    : {summary_path}")
    if capture_path is not None:
        print(f"캡처 CSV    : {capture_path}")


if __name__ == "__main__":
    main()
