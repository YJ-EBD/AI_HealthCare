from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app_paths import append_runtime_log
from metrics import (
    SignalDataset,
    UserProfile,
    bandpass_filter,
    build_average_pulse,
    calculate_blood_pressure_metrics,
    calculate_circulation_metrics,
    calculate_hr_metrics,
    calculate_hrv_metrics,
    calculate_signal_quality,
    calculate_stress_metrics,
    calculate_vascular_age_metrics,
    calculate_vascular_health_metrics,
    clamp,
    detect_beats_from_aux_channel,
    detect_systolic_peaks,
    estimate_sample_rate,
    extract_pulse_features,
    find_onsets,
)


try:
    import neurokit2 as nk  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    nk = None


def build_user_profile(age: int | None, sex: str, calibration_sbp: float | None, calibration_dbp: float | None) -> UserProfile:
    return UserProfile(
        age=age,
        sex=sex or "unknown",
        calibration_sbp=calibration_sbp,
        calibration_dbp=calibration_dbp,
    )


def load_camera_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _analyze_with_neurokit(signal: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    if nk is None or signal.size < 10 or sample_rate_hz <= 0.0:
        return {}
    try:
        cleaned = np.asarray(nk.ppg_clean(signal, sampling_rate=sample_rate_hz), dtype=float)
        _, info = nk.ppg_peaks(cleaned, sampling_rate=sample_rate_hz)
        peaks = np.asarray(info.get("PPG_Peaks") or [], dtype=int)
        if peaks.size < 2:
            return {}
        hrv = nk.hrv_time(info, sampling_rate=sample_rate_hz, show=False)
        result = {
            "heart_rate_bpm": float(60.0 / np.mean(np.diff(peaks) / sample_rate_hz)),
        }
        if len(hrv.index) > 0:
            row = hrv.iloc[0]
            if row.get("HRV_SDNN") is not None:
                result["sdnn_ms"] = float(row["HRV_SDNN"])
            if row.get("HRV_RMSSD") is not None:
                result["rmssd_ms"] = float(row["HRV_RMSSD"])
            if row.get("HRV_pNN50") is not None:
                result["pnn50"] = float(row["HRV_pNN50"])
        return result
    except Exception:  # noqa: BLE001
        return {}


def prepare_context(dataset: SignalDataset) -> dict[str, Any]:
    sample_rate_hz = dataset.sample_rate_hz or estimate_sample_rate(dataset.timestamps_s)
    if sample_rate_hz <= 0.0:
        raise ValueError("샘플링 주파수를 계산하지 못했습니다.")

    warnings: list[str] = []
    raw_ppg = np.asarray(dataset.ppg, dtype=float)
    filtered_ppg = bandpass_filter(raw_ppg, sample_rate_hz, low_hz=0.5, high_hz=4.0)
    filtered_ppg = filtered_ppg - np.mean(filtered_ppg)

    peak_indices = detect_systolic_peaks(filtered_ppg, sample_rate_hz)
    beat_source = "ppg_peak_detection"
    if peak_indices.size < 3:
        fallback_peaks = detect_beats_from_aux_channel(dataset.beat, sample_rate_hz)
        if fallback_peaks.size >= 3:
            peak_indices = fallback_peaks
            beat_source = "beat_channel_fallback"
            warnings.append("PPG 피크 검출이 약해 보조 비트 채널을 대신 사용했습니다.")

    if peak_indices.size < 3:
        raise ValueError("충분한 박동을 검출하지 못했습니다. 센서 상태를 다시 확인해주세요.")

    onset_indices = find_onsets(filtered_ppg, peak_indices, sample_rate_hz)
    pulse_features = extract_pulse_features(filtered_ppg, peak_indices, onset_indices, sample_rate_hz)
    average_pulse = build_average_pulse(filtered_ppg, onset_indices)
    duration_s = float(dataset.timestamps_s[-1] - dataset.timestamps_s[0]) if dataset.timestamps_s.size >= 2 else 0.0
    signal_quality_score = calculate_signal_quality(raw_ppg, filtered_ppg)

    return {
        "sample_rate_hz": float(sample_rate_hz),
        "warnings": warnings,
        "raw_ppg": raw_ppg,
        "filtered_ppg": filtered_ppg,
        "peak_indices": peak_indices,
        "onset_indices": onset_indices,
        "pulse_features": pulse_features,
        "average_pulse": average_pulse,
        "beat_source": beat_source,
        "duration_s": duration_s,
        "signal_quality_score": float(signal_quality_score),
    }


def _apply_neurokit_adjustments(
    context: dict[str, Any],
    heart_rate_metrics: dict[str, Any],
    hrv_metrics: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    neurokit_report = _analyze_with_neurokit(context["raw_ppg"], context["sample_rate_hz"])
    if not neurokit_report:
        return context, heart_rate_metrics, hrv_metrics

    adjusted_context = dict(context)
    adjusted_heart_rate_metrics = dict(heart_rate_metrics)
    adjusted_hrv_metrics = dict(hrv_metrics)

    nk_hr = float(neurokit_report.get("heart_rate_bpm") or 0.0)
    if 40.0 <= nk_hr <= 180.0:
        base_hr = float(adjusted_heart_rate_metrics["heart_rate_bpm"])
        if base_hr > 0.0 and abs(base_hr - nk_hr) <= 12.0:
            adjusted_heart_rate_metrics["heart_rate_bpm"] = float(0.70 * base_hr + 0.30 * nk_hr)
            adjusted_heart_rate_metrics["ibi_mean_ms"] = float(60000.0 / max(adjusted_heart_rate_metrics["heart_rate_bpm"], 1.0))
        elif base_hr <= 0.0:
            adjusted_heart_rate_metrics["heart_rate_bpm"] = nk_hr

    for key in ("sdnn_ms", "rmssd_ms", "pnn50"):
        nk_value = float(neurokit_report.get(key) or 0.0)
        if nk_value > 0.0:
            base_value = float(adjusted_hrv_metrics[key])
            adjusted_hrv_metrics[key] = nk_value if base_value <= 0.0 else float(0.65 * base_value + 0.35 * nk_value)

    adjusted_hrv_metrics["hrv_score"] = float(
        0.35 * clamp((float(adjusted_hrv_metrics["rmssd_ms"]) - 10.0) / 50.0 * 100.0, 0.0, 100.0)
        + 0.30 * clamp((float(adjusted_hrv_metrics["sdnn_ms"]) - 15.0) / 55.0 * 100.0, 0.0, 100.0)
        + 0.15 * clamp(float(adjusted_hrv_metrics["pnn50"]) / 25.0 * 100.0, 0.0, 100.0)
        + 0.20 * clamp(100.0 - abs(float(adjusted_hrv_metrics["lf_hf_ratio"]) - 1.5) / 2.5 * 100.0, 0.0, 100.0)
    )
    warnings.append("NeuroKit2 보조 분석으로 HR/HRV를 미세 보정했습니다.")
    adjusted_context["neurokit2_available"] = True
    return adjusted_context, adjusted_heart_rate_metrics, adjusted_hrv_metrics


def _build_camera_report(
    camera_summary: dict[str, Any] | None,
    ippg_signal_quality_score: float,
    ippg_heart_rate_bpm: float,
    warnings: list[str],
) -> tuple[dict[str, Any], float]:
    report = {
        "available": False,
        "fusion_applied": False,
        "measurement_mode": "ippg_only",
        "measurement_mode_label": "PPG 단독",
        "selected_signal": None,
        "camera_hr_bpm": None,
        "camera_quality_score": 0.0,
        "camera_perfusion_proxy_score": 0.0,
        "camera_perfusion_index_proxy": 0.0,
        "camera_vascular_proxy_score": 0.0,
        "rhythm_stability_score": 0.0,
        "roi_stability_score": 0.0,
        "face_detection_ratio": None,
        "frame_count": 0,
        "heart_rate_difference_bpm": None,
        "camera_weight": 0.0,
        "ippg_weight": 1.0,
    }
    fused_heart_rate_bpm = float(ippg_heart_rate_bpm)

    if not camera_summary:
        return report, fused_heart_rate_bpm

    camera_hr_bpm = float(camera_summary.get("selected_hr_bpm") or 0.0)
    selected_signal_quality_score = float(camera_summary.get("selected_signal_quality_score") or 0.0)
    selected_band_strength_score = float(camera_summary.get("selected_band_strength_score") or 0.0)
    roi_stability_score = float(camera_summary.get("roi_stability_score") or 0.0)
    rhythm_stability_score = float(camera_summary.get("rhythm_stability_score") or 0.0)
    face_detection_ratio = float(camera_summary.get("face_detection_ratio") or 0.0)
    frame_count = int(camera_summary.get("frame_count") or 0)
    camera_quality_score = (
        0.35 * selected_signal_quality_score
        + 0.20 * selected_band_strength_score
        + 0.15 * roi_stability_score
        + 0.15 * rhythm_stability_score
        + 0.15 * (100.0 * clamp(face_detection_ratio, 0.0, 1.0))
    )

    report.update(
        {
            "available": True,
            "measurement_mode": "camera_assisted",
            "measurement_mode_label": "카메라 보조 분석",
            "selected_signal": camera_summary.get("selected_signal"),
            "camera_hr_bpm": camera_hr_bpm if camera_hr_bpm > 0.0 else None,
            "camera_quality_score": float(clamp(camera_quality_score, 0.0, 100.0)),
            "camera_perfusion_proxy_score": float(camera_summary.get("camera_perfusion_proxy_score") or 0.0),
            "camera_perfusion_index_proxy": float(camera_summary.get("camera_perfusion_index_proxy") or 0.0),
            "camera_vascular_proxy_score": float(camera_summary.get("camera_vascular_proxy_score") or 0.0),
            "rhythm_stability_score": rhythm_stability_score,
            "roi_stability_score": roi_stability_score,
            "face_detection_ratio": face_detection_ratio,
            "frame_count": frame_count,
        }
    )

    if not (40.0 <= camera_hr_bpm <= 180.0):
        warnings.append("카메라 심박 추정이 안정적이지 않아 HR 융합에는 사용하지 않았습니다.")
        return report, fused_heart_rate_bpm

    heart_rate_difference_bpm = abs(camera_hr_bpm - ippg_heart_rate_bpm)
    report["heart_rate_difference_bpm"] = float(heart_rate_difference_bpm)
    if report["camera_quality_score"] < 35.0:
        warnings.append("카메라 신호 품질이 낮아 보조 지표만 사용합니다.")
        return report, fused_heart_rate_bpm
    if heart_rate_difference_bpm > 20.0:
        warnings.append("카메라 HR과 PPG HR 차이가 커서 PPG HR을 유지합니다.")
        return report, fused_heart_rate_bpm

    camera_weight = clamp(
        0.20 + 0.20 * (report["camera_quality_score"] / 100.0) - 0.10 * (ippg_signal_quality_score / 100.0),
        0.20,
        0.45,
    )
    ippg_weight = 1.0 - camera_weight
    fused_heart_rate_bpm = float(ippg_weight * ippg_heart_rate_bpm + camera_weight * camera_hr_bpm)
    report.update(
        {
            "fusion_applied": True,
            "measurement_mode": "camera_ppg_fusion",
            "measurement_mode_label": "카메라 + PPG 융합",
            "camera_weight": float(camera_weight),
            "ippg_weight": float(ippg_weight),
        }
    )
    return report, fused_heart_rate_bpm


def _apply_camera_proxy_adjustments(
    camera_report: dict[str, Any],
    circulation_metrics: dict[str, Any],
    vascular_health_metrics: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    adjusted_camera_report = dict(camera_report)
    adjusted_circulation_metrics = dict(circulation_metrics)
    adjusted_vascular_health_metrics = dict(vascular_health_metrics)

    if not adjusted_camera_report.get("available"):
        adjusted_circulation_metrics["camera_weight"] = 0.0
        adjusted_vascular_health_metrics["camera_weight"] = 0.0
        return adjusted_camera_report, adjusted_circulation_metrics, adjusted_vascular_health_metrics

    camera_quality_score = float(adjusted_camera_report.get("camera_quality_score") or 0.0)
    perfusion_proxy = float(adjusted_camera_report.get("camera_perfusion_proxy_score") or 0.0)
    perfusion_index_proxy = float(adjusted_camera_report.get("camera_perfusion_index_proxy") or 0.0)
    vascular_proxy = float(adjusted_camera_report.get("camera_vascular_proxy_score") or 0.0)

    if min(camera_quality_score, perfusion_proxy) >= 35.0:
        circulation_camera_weight = float(clamp(0.10 + 0.15 * (camera_quality_score / 100.0), 0.10, 0.25))
        adjusted_circulation_metrics["circulation_score"] = float(
            (1.0 - circulation_camera_weight) * float(circulation_metrics["circulation_score"])
            + circulation_camera_weight * perfusion_proxy
        )
        adjusted_circulation_metrics["perfusion_index"] = float(
            (1.0 - circulation_camera_weight) * float(circulation_metrics.get("perfusion_index") or 0.0)
            + circulation_camera_weight * perfusion_index_proxy
        )
        adjusted_circulation_metrics["camera_weight"] = circulation_camera_weight
    else:
        adjusted_circulation_metrics["camera_weight"] = 0.0
        warnings.append("카메라 관류 프록시 품질이 낮아 순환 지표는 PPG 중심으로 유지했습니다.")

    if min(camera_quality_score, vascular_proxy) >= 35.0:
        vascular_camera_weight = float(clamp(0.08 + 0.12 * (camera_quality_score / 100.0), 0.08, 0.20))
        adjusted_vascular_health_metrics["vascular_health_score"] = float(
            (1.0 - vascular_camera_weight) * float(vascular_health_metrics["vascular_health_score"])
            + vascular_camera_weight * vascular_proxy
        )
        adjusted_vascular_health_metrics["camera_weight"] = vascular_camera_weight
    else:
        adjusted_vascular_health_metrics["camera_weight"] = 0.0
        warnings.append("카메라 혈관 프록시 품질이 낮아 혈관 건강 지표는 PPG 중심으로 유지했습니다.")

    return adjusted_camera_report, adjusted_circulation_metrics, adjusted_vascular_health_metrics


def _build_quality_report(
    context: dict[str, Any],
    user_profile: UserProfile,
    camera_report: dict[str, Any],
    heart_rate_metrics: dict[str, Any],
    hrv_metrics: dict[str, Any],
    circulation_metrics: dict[str, Any],
    vascular_health_metrics: dict[str, Any],
    vascular_age_metrics: dict[str, Any],
    blood_pressure_metrics: dict[str, Any],
) -> dict[str, Any]:
    signal_quality = float(context["signal_quality_score"])
    duration_s = float(context["duration_s"])
    peak_count = int(context["peak_indices"].size)
    camera_quality = float(camera_report.get("camera_quality_score") or 0.0)

    hr_conf = clamp(0.55 * signal_quality + 0.25 * clamp(peak_count / 45.0 * 100.0, 0.0, 100.0) + 0.20 * camera_quality, 0.0, 100.0)
    hrv_conf = clamp(0.50 * signal_quality + 0.25 * clamp(duration_s / 300.0 * 100.0, 0.0, 100.0) + 0.25 * hr_conf, 0.0, 100.0)
    circulation_conf = clamp(0.65 * signal_quality + 0.20 * float(circulation_metrics["circulation_score"]) + 0.15 * camera_quality, 0.0, 100.0)
    vascular_conf = clamp(0.55 * signal_quality + 0.25 * float(vascular_health_metrics["vascular_health_score"]) + 0.20 * camera_quality, 0.0, 100.0)
    vascular_age_conf = clamp(0.55 * vascular_conf + 0.25 * hrv_conf + 0.20 * (100.0 if user_profile.age is not None else 45.0), 0.0, 100.0)
    bp_conf = clamp(
        0.35 * vascular_conf
        + 0.20 * circulation_conf
        + 0.15 * hr_conf
        + 0.30 * (100.0 if blood_pressure_metrics["calibrated"] else 35.0),
        0.0,
        100.0,
    )

    outputs = {
        "heart_rate": {"confidence_score": hr_conf},
        "hrv": {"confidence_score": hrv_conf},
        "stress": {"confidence_score": hrv_conf},
        "circulation": {"confidence_score": circulation_conf},
        "vascular_health": {"confidence_score": vascular_conf},
        "vascular_age": {"confidence_score": vascular_age_conf},
        "blood_pressure": {"confidence_score": bp_conf},
    }
    no_read_outputs = [name for name, payload in outputs.items() if float(payload["confidence_score"]) < 35.0]
    overall_confidence_score = float(np.mean([float(item["confidence_score"]) for item in outputs.values()]))
    return {
        "outputs": outputs,
        "no_read_outputs": no_read_outputs,
        "overall_confidence_score": overall_confidence_score,
        "input_signal_quality_score": signal_quality,
        "camera_quality_score": camera_quality,
        "peak_count": peak_count,
        "duration_s": duration_s,
        "vascular_age_gap": float(vascular_age_metrics["vascular_age_gap"]),
        "estimated_sbp": float(blood_pressure_metrics["estimated_sbp"]),
        "estimated_dbp": float(blood_pressure_metrics["estimated_dbp"]),
    }


def run_analysis(
    dataset: SignalDataset,
    user_profile: UserProfile,
    camera_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    append_runtime_log("분석 파이프라인을 시작합니다.")
    context = prepare_context(dataset)
    warnings = list(context["warnings"])

    heart_rate_metrics = calculate_hr_metrics(context["peak_indices"], context["sample_rate_hz"])
    hrv_metrics = calculate_hrv_metrics(context["peak_indices"], context["sample_rate_hz"])
    context, heart_rate_metrics, hrv_metrics = _apply_neurokit_adjustments(context, heart_rate_metrics, hrv_metrics, warnings)

    ippg_heart_rate_bpm = float(heart_rate_metrics["heart_rate_bpm"])
    camera_report, fused_heart_rate_bpm = _build_camera_report(
        camera_summary,
        float(context["signal_quality_score"]),
        ippg_heart_rate_bpm,
        warnings,
    )
    if camera_report["fusion_applied"]:
        heart_rate_metrics["heart_rate_bpm"] = fused_heart_rate_bpm
        heart_rate_metrics["ibi_mean_ms"] = float(60000.0 / max(fused_heart_rate_bpm, 1.0))

    stress_metrics = calculate_stress_metrics(float(heart_rate_metrics["heart_rate_bpm"]), hrv_metrics)
    circulation_metrics = calculate_circulation_metrics(context["pulse_features"], context["filtered_ppg"], dataset.aux)
    vascular_health_metrics = calculate_vascular_health_metrics(context["average_pulse"], context["pulse_features"])
    camera_report, circulation_metrics, vascular_health_metrics = _apply_camera_proxy_adjustments(
        camera_report,
        circulation_metrics,
        vascular_health_metrics,
        warnings,
    )
    vascular_age_metrics = calculate_vascular_age_metrics(
        user_profile,
        float(heart_rate_metrics["heart_rate_bpm"]),
        hrv_metrics,
        circulation_metrics,
        vascular_health_metrics,
        context["pulse_features"],
    )
    blood_pressure_metrics = calculate_blood_pressure_metrics(
        user_profile,
        float(heart_rate_metrics["heart_rate_bpm"]),
        circulation_metrics,
        vascular_health_metrics,
        vascular_age_metrics,
        context["pulse_features"],
    )

    if not circulation_metrics["aux_channel_available"]:
        warnings.append("현재 단일 PPG 하드웨어에서는 좌우 채널 차이 항목을 계산하지 않습니다.")
    if not blood_pressure_metrics["calibrated"]:
        warnings.append("개인 보정 혈압값이 없어 혈압은 추세 중심 참고값입니다.")
    if user_profile.age is None:
        warnings.append("나이가 입력되지 않아 혈관 나이는 기본 기준 나이 45세를 사용했습니다.")
    if float(context["signal_quality_score"]) < 35.0:
        warnings.append("신호 품질이 낮습니다. 손가락 압박과 움직임을 줄여주세요.")

    quality_report = _build_quality_report(
        context,
        user_profile,
        camera_report,
        heart_rate_metrics,
        hrv_metrics,
        circulation_metrics,
        vascular_health_metrics,
        vascular_age_metrics,
        blood_pressure_metrics,
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": {
            "sample_rate_hz": float(context["sample_rate_hz"]),
            "sample_count": int(context["raw_ppg"].size),
            "duration_s": float(context["duration_s"]),
            "signal_quality_score": float(context["signal_quality_score"]),
            "beat_detection_source": context["beat_source"],
            "measurement_mode": camera_report["measurement_mode"],
            "measurement_mode_label": camera_report["measurement_mode_label"],
            "neurokit2_available": bool(context.get("neurokit2_available", False)),
        },
        "camera": camera_report,
        "heart_rate": {
            "heart_rate_bpm": float(heart_rate_metrics["heart_rate_bpm"]),
            "heart_rate_ippg_bpm": float(ippg_heart_rate_bpm),
            "heart_rate_camera_bpm": camera_report["camera_hr_bpm"],
            "fusion_applied": bool(camera_report["fusion_applied"]),
            "peak_count": int(context["peak_indices"].size),
            "ibi_mean_ms": float(heart_rate_metrics["ibi_mean_ms"]),
            "ibi_median_ms": float(heart_rate_metrics["ibi_median_ms"]),
        },
        "hrv": hrv_metrics,
        "stress": stress_metrics,
        "circulation": circulation_metrics,
        "vascular_health": vascular_health_metrics,
        "vascular_age": vascular_age_metrics,
        "blood_pressure": blood_pressure_metrics,
        "quality": quality_report,
        "warnings": warnings,
    }
    append_runtime_log("분석 파이프라인이 완료되었습니다.")
    return report


def format_summary_text(report: dict[str, Any]) -> str:
    heart_rate = report["heart_rate"]
    hrv = report["hrv"]
    stress = report["stress"]
    circulation = report["circulation"]
    vascular_health = report["vascular_health"]
    vascular_age = report["vascular_age"]
    blood_pressure = report["blood_pressure"]
    metadata = report["metadata"]
    camera = report.get("camera") or {}
    quality = report.get("quality") or {}
    no_read_outputs = quality.get("no_read_outputs") or []

    lines = [
        "========================================",
        "PSL_Test 심혈관 통합 분석 결과",
        "========================================",
        f"측정 모드      : {metadata.get('measurement_mode_label', 'PPG 단독')}",
        f"신호 품질      : {float(metadata.get('signal_quality_score') or 0.0):.2f} / 100",
        f"전체 신뢰도    : {float(quality.get('overall_confidence_score') or 0.0):.2f} / 100",
        f"1. 심박수      : {heart_rate['heart_rate_bpm']:.2f} bpm",
        f"2. HRV         : SDNN {hrv['sdnn_ms']:.2f} ms | RMSSD {hrv['rmssd_ms']:.2f} ms | pNN50 {hrv['pnn50']:.2f}% | LF/HF {hrv['lf_hf_ratio']:.2f}",
        f"3. 스트레스    : {stress['stress_score']:.2f} / 100 | {stress['stress_state']}",
        f"4. 혈류/순환   : {circulation['circulation_score']:.2f} / 100 | Perfusion {float(circulation.get('perfusion_index') or 0.0):.3f}",
        f"5. 혈관 건강   : {vascular_health['vascular_health_score']:.2f} / 100 | Reflection {vascular_health['reflection_index']}",
        f"6. 혈관 나이   : {vascular_age['vascular_age_estimate']:.1f}세 | 차이 {vascular_age['vascular_age_gap']:+.1f}",
        f"7. 혈압 추정   : {blood_pressure['estimated_sbp']:.1f}/{blood_pressure['estimated_dbp']:.1f} mmHg | {blood_pressure['blood_pressure_trend']}",
        f"무응답 권장    : {', '.join(no_read_outputs) if no_read_outputs else '없음'}",
    ]

    if camera.get("available"):
        camera_hr_text = f"{float(camera['camera_hr_bpm']):.2f} bpm" if camera.get("camera_hr_bpm") is not None else "미검출"
        face_text = f"{float(camera['face_detection_ratio']) * 100.0:.1f}%" if camera.get("face_detection_ratio") is not None else "-"
        lines.append(
            f"카메라 보조    : {camera.get('measurement_mode_label', '카메라 보조')} | 카메라 HR {camera_hr_text} | 얼굴 검출률 {face_text}"
        )

    if report["warnings"]:
        lines.append("경고:")
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines)


def write_report_files(
    output_dir: Path,
    report: dict[str, Any],
    capture_path: Path | None = None,
    extra_paths: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis_report.json"
    summary_path = output_dir / "summary.txt"

    payload = dict(report)
    if capture_path is not None:
        payload["capture_csv"] = str(capture_path)
    if extra_paths:
        payload["extra_paths"] = extra_paths

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(format_summary_text(report))

    append_runtime_log(f"리포트 저장 완료: {report_path}")
    return report_path, summary_path
