from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from cardiovascular_metrics import bandpass_filter, detect_systolic_peaks, estimate_sample_rate
from oss_signal_adapters import analyze_rppg_signal_with_oss


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
    filtered_signal = np.asarray(filtered_signal, dtype=float)
    if filtered_signal.size < 10 or sample_rate_hz <= 0.0:
        return 0.0, 0
    peaks = detect_systolic_peaks(filtered_signal, sample_rate_hz)
    if peaks.size < 3:
        return 0.0, int(peaks.size)
    rr_s = np.diff(peaks) / sample_rate_hz
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
) -> dict[str, object]:
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
    _emit_status(status_callback, f"{video_path.name} 영상에서 카메라 rPPG 특징을 추출합니다.")

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
                _emit_status(status_callback, f"영상 프레임 {frame_index}개를 처리했습니다.")
    finally:
        capture.release()

    if not frame_records:
        raise RuntimeError("처리 가능한 카메라 프레임이 없습니다.")

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
    selected_signal_values = np.asarray(signal_map[selected_signal], dtype=float)
    selected_filtered_values = np.asarray(filtered_map[selected_signal], dtype=float)
    selected_sample_rate_hz = float(estimate_sample_rate(timestamps_array))
    selected_hr_bpm = float(hr_map.get(selected_signal, 0.0))
    selected_hr_bpm_rule_based = float(selected_hr_bpm)
    selected_signal_quality_score = _estimate_signal_quality(selected_signal_values, selected_filtered_values)
    selected_signal_quality_score_rule_based = float(selected_signal_quality_score)
    selected_band_strength_score = float(np.clip(np.std(selected_filtered_values) / 0.35 * 100.0, 0.0, 100.0))
    rhythm_stability_score, selected_peak_count = _estimate_rhythm_stability_score(selected_filtered_values, selected_sample_rate_hz)
    oss_rppg_report = analyze_rppg_signal_with_oss(selected_signal_values, selected_sample_rate_hz)
    selected_signal_quality_score_neurokit = float(oss_rppg_report.get("quality_score") or 0.0)
    selected_peak_count_neurokit = int(oss_rppg_report.get("peak_count") or 0)
    if selected_signal_quality_score_neurokit > 0.0:
        if selected_signal_quality_score_rule_based > 0.0:
            selected_signal_quality_score = float(
                np.clip(
                    0.60 * selected_signal_quality_score_rule_based + 0.40 * selected_signal_quality_score_neurokit,
                    0.0,
                    100.0,
                )
            )
        else:
            selected_signal_quality_score = float(selected_signal_quality_score_neurokit)
    selected_peak_count = max(selected_peak_count, selected_peak_count_neurokit)
    neurokit_hr_bpm = float(oss_rppg_report.get("heart_rate_bpm") or 0.0)
    if 40.0 <= neurokit_hr_bpm <= 180.0:
        if 40.0 <= selected_hr_bpm_rule_based <= 180.0 and abs(neurokit_hr_bpm - selected_hr_bpm_rule_based) <= 18.0:
            selected_hr_bpm = float(0.70 * selected_hr_bpm_rule_based + 0.30 * neurokit_hr_bpm)
        elif not (40.0 <= selected_hr_bpm_rule_based <= 180.0):
            selected_hr_bpm = float(neurokit_hr_bpm)
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
        fieldnames = list(frame_records[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_records:
            writer.writerow(row)

    warnings: list[str] = []
    detection_ratio = detected_face_count / max(len(frame_records), 1)
    if detection_ratio < 0.25:
        warnings.append("얼굴 검출이 약해 중앙 대체 ROI를 자주 사용했습니다.")
    if selected_hr_bpm <= 0.0:
        warnings.append("이 영상에서는 카메라 기반 HR 추정 신뢰도가 낮습니다.")
    for item in oss_rppg_report.get("warnings") or []:
        warnings.append(str(item))
    if oss_rppg_report.get("error"):
        warnings.append(f"NeuroKit2 camera analysis warning: {oss_rppg_report['error']}")

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
        "selected_hr_bpm_rule_based": float(selected_hr_bpm_rule_based),
        "selected_hr_bpm_neurokit": float(neurokit_hr_bpm),
        "selected_hr_bpm": float(selected_hr_bpm),
        "selected_signal_quality_score_rule_based": float(selected_signal_quality_score_rule_based),
        "selected_signal_quality_score_neurokit": float(selected_signal_quality_score_neurokit),
        "selected_signal_quality_score": float(selected_signal_quality_score),
        "selected_band_strength_score": float(selected_band_strength_score),
        "selected_peak_count": int(selected_peak_count),
        "selected_peak_count_neurokit": int(selected_peak_count_neurokit),
        "rhythm_stability_score": float(rhythm_stability_score),
        "roi_stability_score": float(roi_stability_score),
        "camera_perfusion_proxy_score": float(camera_perfusion_proxy_score),
        "camera_perfusion_index_proxy": float(camera_perfusion_index_proxy),
        "camera_vascular_proxy_score": float(camera_vascular_proxy_score),
        "open_source": oss_rppg_report,
        "warnings": warnings,
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _emit_status(status_callback, "카메라 rPPG 특징 추출이 완료되었습니다.")
    return {
        "features_csv_path": features_csv_path,
        "summary_json_path": summary_json_path,
        "selected_signal": selected_signal,
        "selected_hr_bpm": float(selected_hr_bpm),
        "summary": summary,
    }
