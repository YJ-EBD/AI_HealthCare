from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from cardiovascular_metrics import bandpass_filter, detect_systolic_peaks, estimate_sample_rate


EPSILON = 1e-9
StatusCallback = Callable[[str], None] | None


def _emit_status(callback: StatusCallback, message: str) -> None:
    if callback is not None:
        callback(message)


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


def _build_rgb_signals(rgb_trace: np.ndarray) -> dict[str, np.ndarray]:
    if rgb_trace.size == 0:
        empty = np.asarray([], dtype=float)
        return {
            "green": empty,
            "pos": empty,
            "chrom": empty,
        }

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

    return {
        "green": green_signal,
        "pos": pos_signal,
        "chrom": chrom_signal,
    }


def _estimate_hr(signal: np.ndarray, timestamps_s: np.ndarray) -> tuple[float, np.ndarray]:
    sample_rate_hz = estimate_sample_rate(timestamps_s)
    if sample_rate_hz <= 0.0 or signal.size < 10:
        return 0.0, np.asarray([], dtype=float)

    filtered = bandpass_filter(signal, sample_rate_hz, low_hz=0.7, high_hz=3.5)
    peaks = detect_systolic_peaks(filtered, sample_rate_hz)
    if peaks.size < 2:
        return 0.0, filtered
    rr_s = np.diff(peaks) / sample_rate_hz
    if rr_s.size == 0 or np.mean(rr_s) <= 0.0:
        return 0.0, filtered
    return float(60.0 / np.mean(rr_s)), filtered


def extract_camera_rppg_features(
    video_path: Path,
    output_dir: Path,
    frame_timestamps_path: Path | None = None,
    status_callback: StatusCallback = None,
) -> dict[str, Path | float | int | str]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file was not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    features_csv_path = output_dir / "camera_rppg_features.csv"
    summary_json_path = output_dir / "camera_rppg_summary.json"

    timestamp_map = _read_frame_timestamp_map(frame_timestamps_path)
    detector = _create_face_detector()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_records: list[dict[str, float | int | str]] = []
    rgb_trace: list[list[float]] = []
    timestamps_s: list[float] = []
    last_face_box: tuple[int, int, int, int] | None = None
    detected_face_count = 0
    frame_index = 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    _emit_status(status_callback, f"Extracting camera rPPG features from {video_path.name}")

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
                _emit_status(status_callback, f"Processed {frame_index} video frames")
    finally:
        capture.release()

    if not frame_records:
        raise RuntimeError("No usable camera frames were processed.")

    rgb_array = np.asarray(rgb_trace, dtype=float)
    timestamps_array = np.asarray(timestamps_s, dtype=float)
    signal_map = _build_rgb_signals(rgb_array)

    filtered_map: dict[str, np.ndarray] = {}
    hr_map: dict[str, float] = {}
    for signal_name, signal_values in signal_map.items():
        hr_bpm, filtered = _estimate_hr(signal_values, timestamps_array)
        hr_map[signal_name] = hr_bpm
        filtered_map[signal_name] = filtered

    selected_signal = max(hr_map, key=lambda key: abs(hr_map[key])) if hr_map else "green"
    for index, row in enumerate(frame_records):
        row["signal_green"] = float(signal_map["green"][index])
        row["signal_pos"] = float(signal_map["pos"][index])
        row["signal_chrom"] = float(signal_map["chrom"][index])
        row["filtered_green"] = float(filtered_map["green"][index]) if index < filtered_map["green"].size else 0.0
        row["filtered_pos"] = float(filtered_map["pos"][index]) if index < filtered_map["pos"].size else 0.0
        row["filtered_chrom"] = float(filtered_map["chrom"][index]) if index < filtered_map["chrom"].size else 0.0

    with features_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(frame_records[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_records:
            writer.writerow(row)

    warnings: list[str] = []
    detection_ratio = detected_face_count / max(len(frame_records), 1)
    if detection_ratio < 0.25:
        warnings.append("Face detection was weak; center fallback ROI was used often.")
    if hr_map.get(selected_signal, 0.0) <= 0.0:
        warnings.append("Camera HR estimation was weak for this video.")

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
        "selected_hr_bpm": float(hr_map.get(selected_signal, 0.0)),
        "warnings": warnings,
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _emit_status(status_callback, "Camera rPPG feature extraction finished.")
    return {
        "features_csv_path": features_csv_path,
        "summary_json_path": summary_json_path,
        "selected_signal": selected_signal,
        "selected_hr_bpm": float(hr_map.get(selected_signal, 0.0)),
    }
