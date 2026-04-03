from __future__ import annotations

import csv
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import serial

from app_paths import append_runtime_log
from metrics import bandpass_filter, detect_systolic_peaks, estimate_sample_rate
from serial_capture import parse_arduino_line, write_capture_csv


EPSILON = 1e-9
StatusCallback = Callable[[str], None] | None


def _emit_status(callback: StatusCallback, message: str) -> None:
    append_runtime_log(message)
    if callback is not None:
        callback(message)


def _camera_backend_candidates() -> list[int | None]:
    backends: list[int | None] = []
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        backends.append(int(cv2.CAP_DSHOW))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(int(cv2.CAP_MSMF))
    if not backends or os.name != "nt":
        backends.append(None)
    return backends


def _set_capture_property(capture: cv2.VideoCapture, property_id: int | None, value: float | None) -> float | None:
    if property_id is None or value is None:
        return None
    try:
        capture.set(property_id, float(value))
        return float(capture.get(property_id))
    except Exception:  # noqa: BLE001
        return None


def _apply_auto_exposure(capture: cv2.VideoCapture, enabled: bool | None) -> float | None:
    if enabled is None or not hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
        return None
    prop_id = int(cv2.CAP_PROP_AUTO_EXPOSURE)
    candidates = [1.0 if enabled else 0.0, 0.75 if enabled else 0.25]
    for candidate in candidates:
        actual = _set_capture_property(capture, prop_id, candidate)
        if actual is not None:
            return actual
    return None


def _configure_camera(
    capture: cv2.VideoCapture,
    *,
    width: int | None,
    height: int | None,
    fps: float | None,
    auto_exposure: bool | None,
    exposure_value: float | None,
    auto_white_balance: bool | None,
    white_balance_value: float | None,
    gain_value: float | None,
) -> dict[str, float | None]:
    return {
        "width": _set_capture_property(capture, int(cv2.CAP_PROP_FRAME_WIDTH), float(width)) if width else None,
        "height": _set_capture_property(capture, int(cv2.CAP_PROP_FRAME_HEIGHT), float(height)) if height else None,
        "fps": _set_capture_property(capture, int(cv2.CAP_PROP_FPS), float(fps)) if fps else None,
        "auto_exposure": _apply_auto_exposure(capture, auto_exposure),
        "exposure": _set_capture_property(
            capture,
            int(cv2.CAP_PROP_EXPOSURE) if hasattr(cv2, "CAP_PROP_EXPOSURE") else None,
            exposure_value,
        ),
        "auto_white_balance": _set_capture_property(
            capture,
            int(cv2.CAP_PROP_AUTO_WB) if hasattr(cv2, "CAP_PROP_AUTO_WB") else None,
            1.0 if auto_white_balance else 0.0 if auto_white_balance is not None else None,
        ),
        "white_balance": _set_capture_property(
            capture,
            int(cv2.CAP_PROP_WB_TEMPERATURE) if hasattr(cv2, "CAP_PROP_WB_TEMPERATURE") else None,
            white_balance_value,
        ),
        "gain": _set_capture_property(
            capture,
            int(cv2.CAP_PROP_GAIN) if hasattr(cv2, "CAP_PROP_GAIN") else None,
            gain_value,
        ),
    }


def open_camera_capture(
    camera_index: int,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    auto_exposure: bool | None = None,
    exposure_value: float | None = None,
    auto_white_balance: bool | None = None,
    white_balance_value: float | None = None,
    gain_value: float | None = None,
) -> cv2.VideoCapture | None:
    for backend in _camera_backend_candidates():
        capture = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            continue
        _configure_camera(
            capture,
            width=width,
            height=height,
            fps=fps,
            auto_exposure=auto_exposure,
            exposure_value=exposure_value,
            auto_white_balance=auto_white_balance,
            white_balance_value=white_balance_value,
            gain_value=gain_value,
        )
        return capture
    return None


def probe_camera_indices(max_index: int = 5) -> list[dict[str, int | float]]:
    cameras: list[dict[str, int | float]] = []
    for camera_index in range(max_index + 1):
        capture = open_camera_capture(camera_index)
        if capture is None:
            continue
        ok, frame = capture.read()
        if ok and frame is not None:
            cameras.append(
                {
                    "index": camera_index,
                    "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1]),
                    "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0]),
                    "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
                }
            )
        capture.release()
    return cameras


def write_frame_timestamps_csv(path: Path, frame_rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "host_timestamp_s", "relative_host_s", "width", "height"])
        writer.writeheader()
        for row in frame_rows:
            writer.writerow(row)


def capture_multimodal_session(
    port: str,
    baud: int,
    duration_s: float,
    fallback_sample_rate_hz: float,
    output_dir: Path,
    camera_index: int,
    camera_width: int | None,
    camera_height: int | None,
    camera_fps: float | None,
    *,
    camera_auto_exposure: bool | None = True,
    camera_exposure_value: float | None = None,
    camera_auto_white_balance: bool | None = True,
    camera_white_balance_value: float | None = None,
    camera_gain_value: float | None = None,
    camera_retry_count: int = 2,
    serial_retry_count: int = 2,
    reconnect_enabled: bool = True,
    no_data_timeout_s: float = 5.0,
    status_callback: StatusCallback = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_csv_path = output_dir / "capture.csv"
    video_path = output_dir / "camera_rgb.mp4"
    frame_csv_path = output_dir / "camera_frames.csv"
    manifest_path = output_dir / "session_manifest.json"

    stop_event = threading.Event()
    serial_last_error: list[BaseException] = []
    samples: list[dict[str, float | int]] = []
    frame_rows: list[dict[str, float | int]] = []

    serial_attempts_used = 0
    camera_attempts_used = 0

    capture = open_camera_capture(
        camera_index,
        width=camera_width,
        height=camera_height,
        fps=camera_fps,
        auto_exposure=camera_auto_exposure,
        exposure_value=camera_exposure_value,
        auto_white_balance=camera_auto_white_balance,
        white_balance_value=camera_white_balance_value,
        gain_value=camera_gain_value,
    )
    if capture is None:
        raise RuntimeError(f"카메라 {camera_index}를 열 수 없습니다.")

    applied_camera_settings = _configure_camera(
        capture,
        width=camera_width,
        height=camera_height,
        fps=camera_fps,
        auto_exposure=camera_auto_exposure,
        exposure_value=camera_exposure_value,
        auto_white_balance=camera_auto_white_balance,
        white_balance_value=camera_white_balance_value,
        gain_value=camera_gain_value,
    )

    capture_started_at = time.time()
    capture_deadline = capture_started_at + duration_s

    def serial_loop() -> None:
        nonlocal serial_attempts_used
        implicit_index = 0
        while not stop_event.is_set() and time.time() < capture_deadline:
            try:
                with serial.Serial(port=port, baudrate=baud, timeout=0.2) as connection:
                    connection.reset_input_buffer()
                    time.sleep(1.0)
                    last_data_at = time.time()
                    _emit_status(
                        status_callback,
                        f"{port} 포트에서 시리얼 수집 시작 ({serial_attempts_used + 1}/{serial_retry_count + 1}) / {baud} baud",
                    )
                    while not stop_event.is_set() and time.time() < capture_deadline:
                        raw_line = connection.readline()
                        host_timestamp_s = time.time()
                        if not raw_line:
                            if host_timestamp_s - last_data_at >= no_data_timeout_s:
                                raise TimeoutError(f"{no_data_timeout_s:.1f}초 동안 시리얼 데이터가 없습니다.")
                            continue

                        parsed = parse_arduino_line(
                            raw_line.decode("utf-8", errors="ignore"),
                            fallback_sample_rate_hz=fallback_sample_rate_hz,
                            implicit_index=implicit_index,
                        )
                        if parsed is None:
                            continue

                        last_data_at = host_timestamp_s
                        parsed["host_timestamp_s"] = host_timestamp_s
                        parsed["relative_host_s"] = host_timestamp_s - capture_started_at
                        samples.append(parsed)
                        implicit_index += 1
                        if len(samples) % 500 == 0:
                            _emit_status(status_callback, f"시리얼 샘플 {len(samples)}개 수집")
                    return
            except BaseException as exc:  # noqa: BLE001
                serial_last_error[:] = [exc]
                if not reconnect_enabled or serial_attempts_used >= serial_retry_count:
                    stop_event.set()
                    return
                serial_attempts_used += 1
                _emit_status(status_callback, f"시리얼 재연결 시도 ({serial_attempts_used}/{serial_retry_count})")
                time.sleep(1.0)

    serial_thread = threading.Thread(target=serial_loop, name="serial-capture", daemon=True)
    serial_thread.start()

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    failed_frame_reads = 0
    try:
        _emit_status(status_callback, f"카메라 {camera_index} 수집 시작")
        while time.time() < capture_deadline and not stop_event.is_set():
            ok, frame = capture.read()
            host_timestamp_s = time.time()
            if not ok or frame is None:
                failed_frame_reads += 1
                if reconnect_enabled and failed_frame_reads >= 8 and camera_attempts_used < camera_retry_count:
                    camera_attempts_used += 1
                    _emit_status(status_callback, f"카메라 재연결 시도 ({camera_attempts_used}/{camera_retry_count})")
                    capture.release()
                    time.sleep(0.5)
                    capture = open_camera_capture(
                        camera_index,
                        width=camera_width,
                        height=camera_height,
                        fps=camera_fps,
                        auto_exposure=camera_auto_exposure,
                        exposure_value=camera_exposure_value,
                        auto_white_balance=camera_auto_white_balance,
                        white_balance_value=camera_white_balance_value,
                        gain_value=camera_gain_value,
                    )
                    if capture is None:
                        continue
                    applied_camera_settings = _configure_camera(
                        capture,
                        width=camera_width,
                        height=camera_height,
                        fps=camera_fps,
                        auto_exposure=camera_auto_exposure,
                        exposure_value=camera_exposure_value,
                        auto_white_balance=camera_auto_white_balance,
                        white_balance_value=camera_white_balance_value,
                        gain_value=camera_gain_value,
                    )
                    failed_frame_reads = 0
                else:
                    time.sleep(0.02)
                continue

            failed_frame_reads = 0
            height, width = frame.shape[:2]
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_fps = float(camera_fps or capture.get(cv2.CAP_PROP_FPS) or 30.0)
                writer = cv2.VideoWriter(str(video_path), fourcc, max(writer_fps, 1.0), (width, height))
                _emit_status(status_callback, f"{width}x{height} / {writer_fps:.1f} fps 영상 저장 시작")

            writer.write(frame)
            frame_rows.append(
                {
                    "frame_index": frame_count,
                    "host_timestamp_s": host_timestamp_s,
                    "relative_host_s": host_timestamp_s - capture_started_at,
                    "width": width,
                    "height": height,
                }
            )
            frame_count += 1
            if frame_count % 60 == 0:
                _emit_status(status_callback, f"카메라 프레임 {frame_count}개 수집")
    finally:
        stop_event.set()
        capture.release()
        if writer is not None:
            writer.release()
        serial_thread.join(timeout=3.0)

    if serial_last_error and not samples:
        raise RuntimeError(f"시리얼 수집에 실패했습니다: {serial_last_error[0]}")
    if not samples:
        raise RuntimeError("수집된 PPG 샘플이 없습니다.")
    if frame_count == 0:
        raise RuntimeError("수집된 카메라 프레임이 없습니다.")

    write_capture_csv(capture_csv_path, samples)
    write_frame_timestamps_csv(frame_csv_path, frame_rows)

    manifest = {
        "mode": "multimodal_dataset_v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "duration_s": duration_s,
        "stability": {
            "reconnect_enabled": reconnect_enabled,
            "serial_retry_count": serial_retry_count,
            "camera_retry_count": camera_retry_count,
            "no_data_timeout_s": no_data_timeout_s,
            "serial_attempts_used": serial_attempts_used,
            "camera_attempts_used": camera_attempts_used,
        },
        "serial": {
            "port": port,
            "baud": baud,
            "sample_count": len(samples),
        },
        "camera": {
            "index": camera_index,
            "requested_width": camera_width,
            "requested_height": camera_height,
            "requested_fps": camera_fps,
            "applied_settings": applied_camera_settings,
            "frame_count": frame_count,
        },
        "paths": {
            "capture_csv": str(capture_csv_path),
            "camera_video": str(video_path),
            "camera_frames_csv": str(frame_csv_path),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    _emit_status(status_callback, "멀티모달 측정 완료")
    return {
        "capture_csv_path": capture_csv_path,
        "video_path": video_path,
        "frame_csv_path": frame_csv_path,
        "manifest_path": manifest_path,
        "frame_count": frame_count,
        "sample_count": len(samples),
        "camera_settings": applied_camera_settings,
    }


def _read_frame_timestamp_map(path: Path | None) -> dict[int, float]:
    if path is None or not path.exists():
        return {}
    timestamp_map: dict[int, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                frame_index = int(float(row["frame_index"]))
                timestamp_s = float(row["host_timestamp_s"])
            except (KeyError, TypeError, ValueError):
                continue
            timestamp_map[frame_index] = timestamp_s
    return timestamp_map


def _create_face_detector() -> cv2.CascadeClassifier | None:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return None
    detector = cv2.CascadeClassifier(str(cascade_path))
    return detector if not detector.empty() else None


def _largest_face(faces: np.ndarray) -> tuple[int, int, int, int] | None:
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    return int(x), int(y), int(w), int(h)


def _fallback_face_box(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    width = int(frame_width * 0.40)
    height = int(frame_height * 0.50)
    x = max(0, (frame_width - width) // 2)
    y = max(0, int(frame_height * 0.18))
    return x, y, width, height


def _skin_roi_from_face_box(face_box: tuple[int, int, int, int], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x, y, w, h = face_box
    roi_x = int(x + 0.20 * w)
    roi_y = int(y + 0.18 * h)
    roi_w = int(0.60 * w)
    roi_h = int(0.42 * h)
    roi_x = max(0, min(roi_x, frame_width - 1))
    roi_y = max(0, min(roi_y, frame_height - 1))
    roi_w = max(8, min(roi_w, frame_width - roi_x))
    roi_h = max(8, min(roi_h, frame_height - roi_y))
    return roi_x, roi_y, roi_w, roi_h


def _normalize_channel_trace(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=float)
    if trace.size == 0:
        return trace
    centered = trace - np.mean(trace)
    scale = np.std(centered)
    if scale <= EPSILON:
        return np.zeros_like(centered)
    return centered / scale


def _coeff_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    mean_value = float(np.mean(values))
    if abs(mean_value) <= EPSILON:
        return 0.0
    return float(np.std(values) / abs(mean_value))


def _build_rgb_signals(rgb_trace: np.ndarray) -> dict[str, np.ndarray]:
    if rgb_trace.size == 0:
        empty = np.asarray([], dtype=float)
        return {"green": empty, "pos": empty, "chrom": empty}

    rgb_trace = np.asarray(rgb_trace, dtype=float)
    rgb_mean = np.mean(rgb_trace, axis=0)
    rgb_norm = rgb_trace / np.maximum(rgb_mean, EPSILON) - 1.0

    green_signal = _normalize_channel_trace(rgb_trace[:, 1])

    s1_pos = rgb_norm[:, 1] - rgb_norm[:, 2]
    s2_pos = -2.0 * rgb_norm[:, 0] + rgb_norm[:, 1] + rgb_norm[:, 2]
    alpha_pos = float(np.std(s1_pos) / max(np.std(s2_pos), EPSILON))
    pos_signal = _normalize_channel_trace(s1_pos + alpha_pos * s2_pos)

    x_chrom = 3.0 * rgb_norm[:, 0] - 2.0 * rgb_norm[:, 1]
    y_chrom = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
    alpha_chrom = float(np.std(x_chrom) / max(np.std(y_chrom), EPSILON))
    chrom_signal = _normalize_channel_trace(x_chrom - alpha_chrom * y_chrom)

    return {"green": green_signal, "pos": pos_signal, "chrom": chrom_signal}


def _estimate_hr(signal: np.ndarray, timestamps_s: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    sample_rate_hz = estimate_sample_rate(timestamps_s)
    if sample_rate_hz <= 0.0 or signal.size < 10:
        return 0.0, np.asarray([], dtype=float), np.asarray([], dtype=int)
    filtered = bandpass_filter(signal, sample_rate_hz, low_hz=0.7, high_hz=3.5)
    peaks = detect_systolic_peaks(filtered, sample_rate_hz)
    if peaks.size < 2:
        return 0.0, filtered, peaks
    rr_s = np.diff(peaks) / sample_rate_hz
    if rr_s.size == 0 or np.mean(rr_s) <= 0.0:
        return 0.0, filtered, peaks
    return float(60.0 / np.mean(rr_s)), filtered, peaks


def _estimate_signal_quality(raw_signal: np.ndarray, filtered_signal: np.ndarray) -> float:
    raw_signal = np.asarray(raw_signal, dtype=float)
    filtered_signal = np.asarray(filtered_signal, dtype=float)
    if raw_signal.size == 0 or filtered_signal.size == 0:
        return 0.0
    centered_raw = raw_signal - np.mean(raw_signal)
    filtered_std = float(np.std(filtered_signal))
    noise_std = float(np.std(centered_raw[: filtered_signal.size] - filtered_signal))
    return float(np.clip(100.0 * filtered_std / (filtered_std + noise_std + EPSILON), 0.0, 100.0))


def _estimate_rhythm_stability_score(filtered_signal: np.ndarray, sample_rate_hz: float) -> tuple[float, int]:
    if filtered_signal.size < 10 or sample_rate_hz <= 0.0:
        return 0.0, 0
    peaks = detect_systolic_peaks(filtered_signal, sample_rate_hz)
    if peaks.size < 3:
        return 0.0, int(peaks.size)
    rr_s = np.diff(peaks) / sample_rate_hz
    rr_s = rr_s[np.isfinite(rr_s) & (rr_s > 0.0)]
    if rr_s.size == 0 or np.mean(rr_s) <= EPSILON:
        return 0.0, int(peaks.size)
    rr_cv = float(np.std(rr_s) / max(np.mean(rr_s), EPSILON))
    score = float(np.clip(100.0 - (rr_cv / 0.18) * 100.0, 0.0, 100.0))
    return score, int(peaks.size)


def _estimate_roi_stability_score(frame_records: list[dict[str, float | int | str]]) -> float:
    if not frame_records:
        return 0.0
    center_x_values: list[float] = []
    center_y_values: list[float] = []
    area_ratio_values: list[float] = []
    for row in frame_records:
        frame_width = float(row.get("frame_width") or 0.0)
        frame_height = float(row.get("frame_height") or 0.0)
        roi_x = float(row.get("roi_x") or 0.0)
        roi_y = float(row.get("roi_y") or 0.0)
        roi_w = float(row.get("roi_w") or 0.0)
        roi_h = float(row.get("roi_h") or 0.0)
        if frame_width <= 0.0 or frame_height <= 0.0:
            continue
        center_x_values.append((roi_x + 0.5 * roi_w) / frame_width)
        center_y_values.append((roi_y + 0.5 * roi_h) / frame_height)
        area_ratio_values.append((roi_w * roi_h) / max(frame_width * frame_height, EPSILON))
    if not center_x_values or not center_y_values or not area_ratio_values:
        return 0.0
    position_jitter = float(np.hypot(np.std(center_x_values), np.std(center_y_values)))
    area_cv = _coeff_variation(np.asarray(area_ratio_values, dtype=float))
    position_score = float(np.clip(100.0 - (position_jitter / 0.08) * 100.0, 0.0, 100.0))
    area_score = float(np.clip(100.0 - (area_cv / 0.15) * 100.0, 0.0, 100.0))
    return 0.60 * position_score + 0.40 * area_score


def extract_camera_rppg_features(
    video_path: Path,
    output_dir: Path,
    frame_timestamps_path: Path | None = None,
    status_callback: StatusCallback = None,
) -> dict[str, Any]:
    if not video_path.exists():
        raise FileNotFoundError(f"영상 파일을 찾지 못했습니다: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    features_csv_path = output_dir / "camera_rppg_features.csv"
    summary_json_path = output_dir / "camera_rppg_summary.json"

    timestamp_map = _read_frame_timestamp_map(frame_timestamps_path)
    detector = _create_face_detector()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    frame_records: list[dict[str, float | int | str]] = []
    rgb_trace: list[list[float]] = []
    timestamps_s: list[float] = []
    last_face_box: tuple[int, int, int, int] | None = None
    detected_face_count = 0
    frame_index = 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    _emit_status(status_callback, f"{video_path.name} 영상에서 카메라 rPPG 추출 시작")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            frame_height, frame_width = frame.shape[:2]
            if detector is not None and (frame_index % 10 == 0 or last_face_box is None):
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                face_box = _largest_face(faces)
                if face_box is not None:
                    last_face_box = face_box
                    detected_face_count += 1

            if last_face_box is not None:
                face_box = last_face_box
                roi_source = "face"
            else:
                face_box = _fallback_face_box(frame_width, frame_height)
                roi_source = "fallback_center"

            roi_x, roi_y, roi_w, roi_h = _skin_roi_from_face_box(face_box, frame_width, frame_height)
            roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            if roi.size == 0:
                frame_index += 1
                continue

            mean_bgr = np.mean(roi.reshape(-1, 3), axis=0)
            mean_rgb = mean_bgr[::-1]
            rgb_trace.append([float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])])

            if frame_index in timestamp_map:
                timestamp_s = float(timestamp_map[frame_index])
            elif fps > 0.0:
                timestamp_s = frame_index / fps
            else:
                timestamp_s = float(frame_index)
            timestamps_s.append(timestamp_s)

            frame_records.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": timestamp_s,
                    "roi_source": roi_source,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "roi_x": roi_x,
                    "roi_y": roi_y,
                    "roi_w": roi_w,
                    "roi_h": roi_h,
                    "mean_r": float(mean_rgb[0]),
                    "mean_g": float(mean_rgb[1]),
                    "mean_b": float(mean_rgb[2]),
                }
            )

            frame_index += 1
            if frame_index % 120 == 0:
                _emit_status(status_callback, f"영상 프레임 {frame_index}개 처리")
    finally:
        capture.release()

    if not frame_records:
        raise RuntimeError("분석 가능한 카메라 프레임이 없습니다.")

    rgb_array = np.asarray(rgb_trace, dtype=float)
    timestamps_array = np.asarray(timestamps_s, dtype=float)
    signal_map = _build_rgb_signals(rgb_array)

    filtered_map: dict[str, np.ndarray] = {}
    hr_map: dict[str, float] = {}
    peak_count_map: dict[str, int] = {}
    for signal_name, signal_values in signal_map.items():
        hr_bpm, filtered, peaks = _estimate_hr(signal_values, timestamps_array)
        hr_map[signal_name] = hr_bpm
        filtered_map[signal_name] = filtered
        peak_count_map[signal_name] = int(peaks.size)

    selected_signal = max(hr_map, key=lambda key: abs(hr_map[key])) if hr_map else "green"
    selected_signal_values = np.asarray(signal_map[selected_signal], dtype=float)
    selected_filtered_values = np.asarray(filtered_map[selected_signal], dtype=float)
    selected_sample_rate_hz = float(estimate_sample_rate(timestamps_array))
    selected_hr_bpm = float(hr_map.get(selected_signal, 0.0))
    selected_signal_quality_score = _estimate_signal_quality(selected_signal_values, selected_filtered_values)
    selected_band_strength_score = float(np.clip(np.std(selected_filtered_values) / 0.35 * 100.0, 0.0, 100.0))
    rhythm_stability_score, selected_peak_count = _estimate_rhythm_stability_score(selected_filtered_values, selected_sample_rate_hz)
    roi_stability_score = _estimate_roi_stability_score(frame_records)

    camera_perfusion_proxy_score = (
        0.50 * selected_signal_quality_score
        + 0.35 * selected_band_strength_score
        + 0.15 * roi_stability_score
    )
    camera_vascular_proxy_score = (
        0.45 * selected_signal_quality_score
        + 0.35 * rhythm_stability_score
        + 0.20 * roi_stability_score
    )
    camera_perfusion_index_proxy = float(
        np.clip(
            0.50 * (selected_signal_quality_score / 100.0)
            + 0.35 * (selected_band_strength_score / 100.0)
            + 0.15 * (roi_stability_score / 100.0),
            0.0,
            1.0,
        )
    )

    for index, row in enumerate(frame_records):
        row["signal_green"] = float(signal_map["green"][index])
        row["signal_pos"] = float(signal_map["pos"][index])
        row["signal_chrom"] = float(signal_map["chrom"][index])
        row["filtered_green"] = float(filtered_map["green"][index]) if index < filtered_map["green"].size else 0.0
        row["filtered_pos"] = float(filtered_map["pos"][index]) if index < filtered_map["pos"].size else 0.0
        row["filtered_chrom"] = float(filtered_map["chrom"][index]) if index < filtered_map["chrom"].size else 0.0

    with features_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_records[0].keys()))
        writer.writeheader()
        for row in frame_records:
            writer.writerow(row)

    warnings: list[str] = []
    detection_ratio = detected_face_count / max(len(frame_records), 1)
    if detection_ratio < 0.25:
        warnings.append("얼굴 검출이 약해 중앙 대체 ROI를 자주 사용했습니다.")
    if selected_hr_bpm <= 0.0:
        warnings.append("카메라 기반 심박 추정이 약합니다.")

    summary = {
        "video_path": str(video_path),
        "frame_timestamps_path": str(frame_timestamps_path) if frame_timestamps_path is not None else None,
        "frame_count": len(frame_records),
        "sample_rate_hz": float(estimate_sample_rate(timestamps_array)),
        "face_detection_ratio": float(detection_ratio),
        "hr_green_bpm": float(hr_map["green"]),
        "hr_pos_bpm": float(hr_map["pos"]),
        "hr_chrom_bpm": float(hr_map["chrom"]),
        "selected_signal": selected_signal,
        "selected_hr_bpm": float(selected_hr_bpm),
        "selected_signal_quality_score": float(selected_signal_quality_score),
        "selected_band_strength_score": float(selected_band_strength_score),
        "selected_peak_count": int(selected_peak_count),
        "rhythm_stability_score": float(rhythm_stability_score),
        "roi_stability_score": float(roi_stability_score),
        "camera_perfusion_proxy_score": float(camera_perfusion_proxy_score),
        "camera_perfusion_index_proxy": float(camera_perfusion_index_proxy),
        "camera_vascular_proxy_score": float(camera_vascular_proxy_score),
        "warnings": warnings,
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    _emit_status(status_callback, "카메라 rPPG 추출 완료")
    return {
        "features_csv_path": features_csv_path,
        "summary_json_path": summary_json_path,
        "selected_signal": selected_signal,
        "selected_hr_bpm": float(selected_hr_bpm),
        "summary": summary,
    }
