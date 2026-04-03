from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import serial
from serial.tools import list_ports

from app_paths import append_runtime_log
from metrics import SignalDataset, estimate_sample_rate


StatusCallback = Callable[[str], None] | None


def list_serial_ports() -> list[dict[str, str]]:
    ports: list[dict[str, str]] = []
    for port in list_ports.comports():
        ports.append(
            {
                "device": str(port.device),
                "description": str(port.description or port.hwid or port.device),
            }
        )
    return ports


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


def _emit_status(callback: StatusCallback, message: str) -> None:
    append_runtime_log(message)
    if callback is not None:
        callback(message)


def capture_serial_session(
    port: str,
    baud: int,
    duration_s: float,
    fallback_sample_rate_hz: float,
    status_callback: StatusCallback = None,
    retry_count: int = 2,
    reopen_delay_s: float = 1.0,
    no_data_timeout_s: float = 5.0,
) -> list[dict[str, float | int]]:
    last_error: BaseException | None = None

    for attempt in range(retry_count + 1):
        samples: list[dict[str, float | int]] = []
        started_at = time.time()
        last_data_at = started_at
        attempt_label = f"{attempt + 1}/{retry_count + 1}"

        try:
            _emit_status(
                status_callback,
                f"{port} 포트에서 {baud} baud로 {duration_s:.1f}초 측정을 시작합니다. (시도 {attempt_label})",
            )
            with serial.Serial(port=port, baudrate=baud, timeout=0.3) as connection:
                connection.reset_input_buffer()
                time.sleep(1.0)

                while time.time() - started_at < duration_s:
                    raw_line = connection.readline()
                    now = time.time()
                    if not raw_line:
                        if now - last_data_at >= no_data_timeout_s:
                            raise TimeoutError(f"{no_data_timeout_s:.1f}초 동안 시리얼 데이터가 들어오지 않았습니다.")
                        continue

                    line = raw_line.decode("utf-8", errors="ignore")
                    parsed = parse_arduino_line(line, fallback_sample_rate_hz=fallback_sample_rate_hz, implicit_index=len(samples))
                    if parsed is None:
                        continue

                    last_data_at = now
                    samples.append(parsed)
                    if len(samples) % 500 == 0:
                        elapsed = now - started_at
                        _emit_status(status_callback, f"PPG 샘플 {len(samples)}개 수집 완료 ({elapsed:.1f}초)")

            if samples:
                _emit_status(status_callback, f"시리얼 수집 완료: 총 {len(samples)} 샘플")
                return samples
            raise RuntimeError("수집된 샘플이 없습니다.")
        except BaseException as exc:  # noqa: BLE001
            last_error = exc
            error_text = str(exc)
            if "PermissionError" in error_text or "액세스가 거부되었습니다" in error_text:
                error_text = f"{exc} | 다른 프로그램이 이 포트를 사용 중일 수 있습니다. Arduino IDE Serial Monitor, 다른 PSL_Test 창, 터미널을 닫아주세요."
            _emit_status(status_callback, f"시리얼 수집 실패: {error_text}")
            if attempt >= retry_count:
                break
            time.sleep(reopen_delay_s)

    last_error_text = str(last_error)
    if "PermissionError" in last_error_text or "액세스가 거부되었습니다" in last_error_text:
        last_error_text = (
            f"{last_error} | 다른 프로그램이 포트를 사용 중일 수 있습니다. "
            "Arduino IDE Serial Monitor, 다른 PSL_Test 창, 시리얼 터미널을 닫은 뒤 다시 시도하세요."
        )
    raise RuntimeError(f"시리얼 수집에 실패했습니다: {last_error_text}") from last_error


def write_capture_csv(path: Path, samples: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    standard_fieldnames = ["timestamp_s", "sample", "ppg", "beat", "ppg_raw", "beat_raw", "ppg_v", "beat_v"]
    extra_fieldnames: list[str] = []
    for sample in samples:
        for key in sample:
            if key not in standard_fieldnames and key not in extra_fieldnames:
                extra_fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*standard_fieldnames, *extra_fieldnames])
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
        raise ValueError(f"{path} 파일에 데이터가 없습니다.")

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
            raise ValueError("CSV에서 사용할 수 있는 PPG 컬럼을 찾지 못했습니다.")

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
    sample_rate_hz = estimate_sample_rate(timestamp_array) or fallback_sample_rate_hz

    return SignalDataset(
        timestamps_s=timestamp_array,
        ppg=ppg_array,
        beat=beat_array,
        aux=aux_array,
        sample_rate_hz=float(sample_rate_hz),
    )
