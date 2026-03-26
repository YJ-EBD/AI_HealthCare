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


def open_camera_capture(camera_index: int) -> cv2.VideoCapture | None:
    for backend in _camera_backend_candidates():
        capture = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
        if capture.isOpened():
            return capture
        capture.release()
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


def _configure_camera(
    capture: cv2.VideoCapture,
    width: int | None,
    height: int | None,
    fps: float | None,
) -> None:
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps:
        capture.set(cv2.CAP_PROP_FPS, float(fps))


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
    status_callback: StatusCallback = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_csv_path = output_dir / "capture.csv"
    video_path = output_dir / "camera_rgb.mp4"
    frame_csv_path = output_dir / "camera_frames.csv"
    manifest_path = output_dir / "session_manifest.json"

    capture_started_at = time.time()
    stop_event = threading.Event()
    serial_error: list[BaseException] = []
    samples: list[dict[str, float | int]] = []
    frame_rows: list[dict[str, float | int]] = []

    def serial_loop() -> None:
        implicit_index = 0
        try:
            with serial.Serial(port=port, baudrate=baud, timeout=0.2) as connection:
                connection.reset_input_buffer()
                time.sleep(1.0)
                _emit_status(status_callback, f"Serial capture started on {port} at {baud} baud.")
                while not stop_event.is_set():
                    raw_line = connection.readline()
                    host_timestamp_s = time.time()
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="ignore")
                    parsed = parse_arduino_line(
                        line,
                        fallback_sample_rate_hz=fallback_sample_rate_hz,
                        implicit_index=implicit_index,
                    )
                    if parsed is None:
                        continue
                    parsed["host_timestamp_s"] = host_timestamp_s
                    parsed["relative_host_s"] = host_timestamp_s - capture_started_at
                    samples.append(parsed)
                    implicit_index += 1
                    if len(samples) % 500 == 0:
                        _emit_status(status_callback, f"Serial samples collected: {len(samples)}")
        except BaseException as exc:  # noqa: BLE001
            serial_error.append(exc)
        finally:
            stop_event.set()

    serial_thread = threading.Thread(target=serial_loop, name="serial-capture", daemon=True)
    serial_thread.start()

    capture = open_camera_capture(camera_index)
    if capture is None:
        stop_event.set()
        serial_thread.join(timeout=2.0)
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    writer: cv2.VideoWriter | None = None
    frame_count = 0
    try:
        _configure_camera(capture, width=camera_width, height=camera_height, fps=camera_fps)
        _emit_status(status_callback, f"Camera capture started on index {camera_index}.")
        while time.time() - capture_started_at < duration_s and not stop_event.is_set():
            ok, frame = capture.read()
            host_timestamp_s = time.time()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            height, width = frame.shape[:2]
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_fps = float(camera_fps or capture.get(cv2.CAP_PROP_FPS) or 30.0)
                writer = cv2.VideoWriter(str(video_path), fourcc, max(writer_fps, 1.0), (width, height))
                _emit_status(status_callback, f"Video writer opened at {width}x{height} @ {writer_fps:.1f} fps.")

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
                _emit_status(status_callback, f"Camera frames collected: {frame_count}")
    finally:
        stop_event.set()
        capture.release()
        if writer is not None:
            writer.release()
        serial_thread.join(timeout=3.0)

    if serial_error:
        raise RuntimeError(f"Serial capture failed: {serial_error[0]}")
    if not samples:
        raise RuntimeError("No usable PPG samples were captured during multimodal recording.")
    if frame_count == 0:
        raise RuntimeError("No camera frames were captured during multimodal recording.")

    write_capture_csv(capture_csv_path, samples)
    write_frame_timestamps_csv(frame_csv_path, frame_rows)

    manifest = {
        "mode": "multimodal_dataset_v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "duration_s": duration_s,
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
            "frame_count": frame_count,
        },
        "paths": {
            "capture_csv": str(capture_csv_path),
            "camera_video": str(video_path),
            "camera_frames_csv": str(frame_csv_path),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    _emit_status(status_callback, "Multimodal capture session saved.")
    return {
        "capture_csv_path": capture_csv_path,
        "video_path": video_path,
        "frame_csv_path": frame_csv_path,
        "manifest_path": manifest_path,
        "frame_count": frame_count,
        "sample_count": len(samples),
    }
