from __future__ import annotations

import csv
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import serial

from capture_and_analyze import parse_arduino_line, write_capture_csv
from runtime_support import log_event, log_exception


StatusCallback = Callable[[str], None] | None


def _emit_status(callback: StatusCallback, message: str) -> None:
    if callback is not None:
        callback(message)


def _camera_backend_candidates() -> list[int | None]:
    backends: list[int | None] = []
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        backends.append(int(cv2.CAP_DSHOW))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(int(cv2.CAP_MSMF))
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
    applied = {
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
    return applied


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
    fieldnames = ["frame_index", "host_timestamp_s", "relative_host_s", "width", "height"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    camera_auto_exposure: bool | None = None,
    camera_exposure_value: float | None = None,
    camera_auto_white_balance: bool | None = None,
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

    capture_started_at = time.time()
    capture_deadline = capture_started_at + duration_s
    stop_event = threading.Event()
    serial_last_error: list[BaseException] = []
    samples: list[dict[str, float | int]] = []
    frame_rows: list[dict[str, float | int]] = []

    serial_attempts_used = 0
    camera_attempts_used = 0

    def serial_loop() -> None:
        nonlocal serial_attempts_used
        implicit_index = 0
        while not stop_event.is_set() and time.time() < capture_deadline:
            try:
                log_event(
                    "multimodal_serial",
                    "멀티모달 시리얼 수집을 시작합니다.",
                    details={"port": port, "baud": baud, "attempt": serial_attempts_used + 1},
                )
                with serial.Serial(port=port, baudrate=baud, timeout=0.2) as connection:
                    connection.reset_input_buffer()
                    time.sleep(1.0)
                    last_data_at = time.time()
                    _emit_status(status_callback, f"{port} 포트에서 시리얼 수집을 시작합니다. (시도 {serial_attempts_used + 1}/{serial_retry_count + 1})")
                    while not stop_event.is_set() and time.time() < capture_deadline:
                        raw_line = connection.readline()
                        host_timestamp_s = time.time()
                        if not raw_line:
                            if host_timestamp_s - last_data_at >= no_data_timeout_s:
                                raise TimeoutError(f"{no_data_timeout_s:.1f}초 동안 시리얼 데이터가 없습니다.")
                            continue
                        line = raw_line.decode("utf-8", errors="ignore")
                        parsed = parse_arduino_line(
                            line,
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
                            _emit_status(status_callback, f"시리얼 샘플 수집 {len(samples)}")
                    return
            except BaseException as exc:  # noqa: BLE001
                serial_last_error[:] = [exc]
                log_exception(
                    "multimodal_serial",
                    exc,
                    details={"port": port, "attempt": serial_attempts_used + 1, "retry_limit": serial_retry_count},
                )
                if not reconnect_enabled or serial_attempts_used >= serial_retry_count:
                    stop_event.set()
                    return
                serial_attempts_used += 1
                _emit_status(status_callback, f"시리얼 연결을 다시 시도합니다. ({serial_attempts_used}/{serial_retry_count})")
                time.sleep(1.0)

    serial_thread = threading.Thread(target=serial_loop, name="serial-capture", daemon=True)
    serial_thread.start()

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
        stop_event.set()
        serial_thread.join(timeout=2.0)
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

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    failed_frame_reads = 0
    try:
        _emit_status(status_callback, f"카메라 {camera_index} 수집을 시작합니다.")
        while time.time() < capture_deadline and not stop_event.is_set():
            ok, frame = capture.read()
            host_timestamp_s = time.time()
            if not ok or frame is None:
                failed_frame_reads += 1
                if reconnect_enabled and failed_frame_reads >= 8 and camera_attempts_used < camera_retry_count:
                    camera_attempts_used += 1
                    _emit_status(status_callback, f"카메라 프레임이 끊겨 재연결합니다. ({camera_attempts_used}/{camera_retry_count})")
                    log_event(
                        "multimodal_camera",
                        "카메라 재연결을 시도합니다.",
                        details={"camera_index": camera_index, "attempt": camera_attempts_used},
                    )
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
                _emit_status(status_callback, f"{width}x{height} / {writer_fps:.1f} fps 설정으로 영상 저장을 시작합니다.")

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
                _emit_status(status_callback, f"카메라 프레임 수집 {frame_count}")
    except BaseException as exc:  # noqa: BLE001
        log_exception("multimodal_camera", exc, details={"camera_index": camera_index})
        raise
    finally:
        stop_event.set()
        capture.release()
        if writer is not None:
            writer.release()
        serial_thread.join(timeout=3.0)

    if serial_last_error and not samples:
        raise RuntimeError(f"시리얼 수집에 실패했습니다: {serial_last_error[0]}")
    if not samples:
        raise RuntimeError("멀티모달 측정 중 사용할 수 있는 PPG 샘플을 수집하지 못했습니다.")
    if frame_count == 0:
        raise RuntimeError("멀티모달 측정 중 카메라 프레임을 수집하지 못했습니다.")

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
            "requested_auto_exposure": camera_auto_exposure,
            "requested_exposure": camera_exposure_value,
            "requested_auto_white_balance": camera_auto_white_balance,
            "requested_white_balance": camera_white_balance_value,
            "requested_gain": camera_gain_value,
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

    log_event(
        "multimodal_capture",
        "멀티모달 세션이 완료되었습니다.",
        details={
            "sample_count": len(samples),
            "frame_count": frame_count,
            "manifest_path": str(manifest_path),
        },
    )
    _emit_status(status_callback, "멀티모달 측정 세션이 완료되었습니다.")
    return {
        "capture_csv_path": capture_csv_path,
        "video_path": video_path,
        "frame_csv_path": frame_csv_path,
        "manifest_path": manifest_path,
        "frame_count": frame_count,
        "sample_count": len(samples),
        "camera_settings": applied_camera_settings,
    }
