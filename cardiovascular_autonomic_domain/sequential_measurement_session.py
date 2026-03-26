from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Callable

import numpy as np

from capture_and_analyze import (
    build_user_profile,
    capture_serial_session,
    load_dataset_from_csv,
    write_capture_csv,
    write_report_files,
)
from cardiovascular_metrics import (
    SignalDataset,
    UserProfile,
    bandpass_filter,
    build_average_pulse,
    calculate_blood_pressure_metrics,
    calculate_circulation_metrics,
    clamp,
    calculate_hr_metrics,
    calculate_hrv_metrics,
    calculate_signal_quality,
    calculate_stress_metrics,
    calculate_vascular_age_metrics,
    calculate_vascular_health_metrics,
    detect_beats_from_aux_channel,
    detect_systolic_peaks,
    estimate_sample_rate,
    extract_pulse_features,
    find_onsets,
)
from ml_inference import build_feature_payload, run_ml_inference
from oss_signal_adapters import analyze_contact_ppg_with_oss


def translate_stress_state(value: str) -> str:
    return {
        "stable": "안정",
        "normal": "보통",
        "tense": "긴장",
    }.get(value, value)


def translate_bp_trend(value: str) -> str:
    return {
        "rising": "상승 추세",
        "falling": "하강 추세",
        "stable": "안정 추세",
    }.get(value, value)


def translate_step_label(value: str) -> str:
    return {
        "1.1 Heart rate": "1.1 심박수",
        "1.2 HRV": "1.2 HRV",
        "1.3 Stress": "1.3 스트레스",
        "1.4 Circulation": "1.4 순환",
        "1.5 Vascular health": "1.5 혈관 건강",
        "1.6 Vascular age": "1.6 혈관 나이",
        "1.7 Blood pressure": "1.7 혈압",
    }.get(value, value)


def translate_camera_signal_name(value: str) -> str:
    return {
        "green": "녹색 채널",
        "pos": "POS",
        "chrom": "CHROM",
    }.get(value, value)


def translate_camera_warning(value: str) -> str:
    mapping = {
        "Face detection was weak; center fallback ROI was used often.": "얼굴 검출이 약해 중앙 대체 ROI를 자주 사용했습니다.",
        "Camera HR estimation was weak for this video.": "이 영상에서는 카메라 기반 HR 추정 신뢰도가 낮습니다.",
    }
    return mapping.get(value, value)


def load_camera_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


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
        "measurement_mode_label": "iPPG 단독",
        "selected_signal": None,
        "selected_signal_label": None,
        "camera_hr_bpm": None,
        "camera_quality_score": 0.0,
        "selected_signal_quality_score": 0.0,
        "selected_band_strength_score": 0.0,
        "camera_perfusion_proxy_score": 0.0,
        "camera_perfusion_index_proxy": 0.0,
        "camera_vascular_proxy_score": 0.0,
        "rhythm_stability_score": 0.0,
        "roi_stability_score": 0.0,
        "face_detection_ratio": None,
        "frame_count": 0,
        "selected_peak_count": 0,
        "heart_rate_difference_bpm": None,
        "fusion_reason": "카메라 데이터가 없어 iPPG만 사용했습니다.",
        "camera_weight": 0.0,
        "ippg_weight": 1.0,
        "proxy_metrics_available": False,
        "circulation_fusion_applied": False,
        "vascular_fusion_applied": False,
    }
    fused_heart_rate_bpm = float(ippg_heart_rate_bpm)

    if not camera_summary:
        return report, fused_heart_rate_bpm

    camera_hr_bpm = float(camera_summary.get("selected_hr_bpm") or 0.0)
    face_detection_ratio = float(camera_summary.get("face_detection_ratio") or 0.0)
    frame_count = int(camera_summary.get("frame_count") or 0)
    selected_signal = str(camera_summary.get("selected_signal") or "")
    selected_signal_quality_score = float(camera_summary.get("selected_signal_quality_score") or 0.0)
    selected_band_strength_score = float(camera_summary.get("selected_band_strength_score") or 0.0)
    camera_perfusion_proxy_score = float(camera_summary.get("camera_perfusion_proxy_score") or 0.0)
    camera_perfusion_index_proxy = float(camera_summary.get("camera_perfusion_index_proxy") or 0.0)
    camera_vascular_proxy_score = float(camera_summary.get("camera_vascular_proxy_score") or 0.0)
    rhythm_stability_score = float(camera_summary.get("rhythm_stability_score") or 0.0)
    roi_stability_score = float(camera_summary.get("roi_stability_score") or 0.0)
    selected_peak_count = int(camera_summary.get("selected_peak_count") or 0)
    camera_warnings = [translate_camera_warning(str(item)) for item in camera_summary.get("warnings") or []]

    report.update(
        {
            "available": True,
            "measurement_mode": "camera_assisted_ippg_only",
            "measurement_mode_label": "카메라 보조 수집 / iPPG 분석",
            "selected_signal": selected_signal or None,
            "selected_signal_label": translate_camera_signal_name(selected_signal) if selected_signal else None,
            "camera_hr_bpm": float(camera_hr_bpm) if camera_hr_bpm > 0.0 else None,
            "selected_signal_quality_score": float(selected_signal_quality_score),
            "selected_band_strength_score": float(selected_band_strength_score),
            "camera_perfusion_proxy_score": float(camera_perfusion_proxy_score),
            "camera_perfusion_index_proxy": float(camera_perfusion_index_proxy),
            "camera_vascular_proxy_score": float(camera_vascular_proxy_score),
            "rhythm_stability_score": float(rhythm_stability_score),
            "roi_stability_score": float(roi_stability_score),
            "face_detection_ratio": float(face_detection_ratio),
            "frame_count": frame_count,
            "selected_peak_count": selected_peak_count,
            "proxy_metrics_available": any(
                key in camera_summary
                for key in (
                    "selected_signal_quality_score",
                    "selected_band_strength_score",
                    "camera_perfusion_proxy_score",
                    "camera_vascular_proxy_score",
                )
            ),
        }
    )

    valid_hr = 40.0 <= camera_hr_bpm <= 180.0
    if report["proxy_metrics_available"]:
        quality_score = (
            0.35 * selected_signal_quality_score
            + 0.20 * selected_band_strength_score
            + 0.15 * roi_stability_score
            + 0.10 * rhythm_stability_score
            + 0.10 * (100.0 * clamp(face_detection_ratio, 0.0, 1.0))
            + 0.10 * (100.0 * clamp(frame_count / 300.0, 0.0, 1.0))
        )
    else:
        quality_score = 0.0
        if valid_hr:
            quality_score += 30.0
            quality_score += 40.0 * clamp(face_detection_ratio, 0.0, 1.0)
            quality_score += 20.0 * clamp(frame_count / 300.0, 0.0, 1.0)
            quality_score += 10.0 * clamp(1.0 - (len(camera_warnings) / 3.0), 0.0, 1.0)
    if not valid_hr:
        quality_score *= 0.6
    report["camera_quality_score"] = float(clamp(quality_score, 0.0, 100.0))

    if camera_warnings:
        warnings.extend(f"카메라: {item}" for item in camera_warnings)

    if not valid_hr:
        warnings.append("카메라 심박 추정이 안정적이지 않아 최종 결과에는 iPPG만 사용했습니다.")
        report["fusion_reason"] = "카메라 심박 추정이 안정적이지 않아 iPPG만 사용했습니다."
        return report, fused_heart_rate_bpm

    heart_rate_difference_bpm = abs(camera_hr_bpm - ippg_heart_rate_bpm)
    report["heart_rate_difference_bpm"] = float(heart_rate_difference_bpm)

    if report["camera_quality_score"] < 35.0:
        warnings.append("카메라 신호 품질이 낮아 최종 결과에는 iPPG만 사용했습니다.")
        report["fusion_reason"] = "카메라 신호 품질이 낮아 iPPG만 사용했습니다."
        return report, fused_heart_rate_bpm

    if heart_rate_difference_bpm > 20.0:
        warnings.append("카메라 HR과 iPPG HR 차이가 커서 최종 결과에는 iPPG만 사용했습니다.")
        report["fusion_reason"] = "카메라 HR과 iPPG HR 차이가 커서 iPPG만 사용했습니다."
        return report, fused_heart_rate_bpm

    agreement_score = clamp(100.0 - (heart_rate_difference_bpm / 12.0) * 100.0, 0.0, 100.0)
    camera_weight = clamp(
        0.20
        + 0.20 * (report["camera_quality_score"] / 100.0)
        + 0.10 * (agreement_score / 100.0)
        - 0.10 * (clamp(ippg_signal_quality_score, 0.0, 100.0) / 100.0),
        0.20,
        0.45,
    )
    ippg_weight = 1.0 - camera_weight
    fused_heart_rate_bpm = float(ippg_weight * ippg_heart_rate_bpm + camera_weight * camera_hr_bpm)

    report.update(
        {
            "fusion_applied": True,
            "measurement_mode": "camera_ippg_fusion",
            "measurement_mode_label": "카메라 + iPPG 융합",
            "fusion_reason": "카메라 HR과 iPPG HR이 잘 일치해 융합 결과를 사용했습니다.",
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

    adjusted_circulation_metrics["circulation_score_ippg"] = float(circulation_metrics["circulation_score"])
    adjusted_vascular_health_metrics["vascular_health_score_ippg"] = float(vascular_health_metrics["vascular_health_score"])

    if not adjusted_camera_report.get("available") or not adjusted_camera_report.get("proxy_metrics_available"):
        return adjusted_camera_report, adjusted_circulation_metrics, adjusted_vascular_health_metrics

    camera_quality_score = float(adjusted_camera_report.get("camera_quality_score") or 0.0)
    camera_perfusion_proxy_score = float(adjusted_camera_report.get("camera_perfusion_proxy_score") or 0.0)
    camera_perfusion_index_proxy = float(adjusted_camera_report.get("camera_perfusion_index_proxy") or 0.0)
    camera_vascular_proxy_score = float(adjusted_camera_report.get("camera_vascular_proxy_score") or 0.0)

    adjusted_circulation_metrics["circulation_score_camera"] = float(camera_perfusion_proxy_score)
    adjusted_circulation_metrics["camera_perfusion_index_proxy"] = float(camera_perfusion_index_proxy)
    adjusted_vascular_health_metrics["vascular_health_score_camera"] = float(camera_vascular_proxy_score)

    circulation_quality_gate = min(camera_quality_score, camera_perfusion_proxy_score)
    if circulation_quality_gate >= 35.0 and camera_perfusion_proxy_score > 0.0:
        circulation_camera_weight = float(clamp(0.12 + 0.18 * (camera_quality_score / 100.0), 0.12, 0.30))
        adjusted_circulation_metrics["circulation_score"] = float(
            (1.0 - circulation_camera_weight) * float(circulation_metrics["circulation_score"])
            + circulation_camera_weight * camera_perfusion_proxy_score
        )
        adjusted_circulation_metrics["perfusion_index"] = float(
            (1.0 - circulation_camera_weight) * float(circulation_metrics.get("perfusion_index", 0.0) or 0.0)
            + circulation_camera_weight * camera_perfusion_index_proxy
        )
        adjusted_circulation_metrics["camera_weight"] = circulation_camera_weight
        adjusted_camera_report["circulation_fusion_applied"] = True
    else:
        adjusted_circulation_metrics["camera_weight"] = 0.0
        warnings.append("카메라 관류 프록시 품질이 낮아 1.4 순환 결과는 iPPG 중심으로 유지했습니다.")

    vascular_quality_gate = min(camera_quality_score, camera_vascular_proxy_score)
    if vascular_quality_gate >= 35.0 and camera_vascular_proxy_score > 0.0:
        vascular_camera_weight = float(clamp(0.10 + 0.15 * (camera_quality_score / 100.0), 0.10, 0.25))
        adjusted_vascular_health_metrics["vascular_health_score"] = float(
            (1.0 - vascular_camera_weight) * float(vascular_health_metrics["vascular_health_score"])
            + vascular_camera_weight * camera_vascular_proxy_score
        )
        adjusted_vascular_health_metrics["camera_weight"] = vascular_camera_weight
        adjusted_camera_report["vascular_fusion_applied"] = True
    else:
        adjusted_vascular_health_metrics["camera_weight"] = 0.0
        warnings.append("카메라 혈관 프록시 품질이 낮아 1.5 혈관 건강 결과는 iPPG 중심으로 유지했습니다.")

    if not adjusted_camera_report["fusion_applied"] and (
        adjusted_camera_report["circulation_fusion_applied"] or adjusted_camera_report["vascular_fusion_applied"]
    ):
        adjusted_camera_report["measurement_mode"] = "camera_assisted_multimodal"
        adjusted_camera_report["measurement_mode_label"] = "카메라 보조 멀티모달 분석"

    return adjusted_camera_report, adjusted_circulation_metrics, adjusted_vascular_health_metrics


def _recalculate_hrv_support_metrics(
    heart_rate_bpm: float,
    sdnn_ms: float,
    rmssd_ms: float,
    pnn50: float,
) -> dict[str, float | str]:
    rmssd_score = clamp((rmssd_ms - 10.0) / 50.0 * 100.0, 0.0, 100.0)
    sdnn_score = clamp((sdnn_ms - 15.0) / 55.0 * 100.0, 0.0, 100.0)
    pnn50_score = clamp(pnn50 / 25.0 * 100.0, 0.0, 100.0)
    hrv_score = 0.45 * rmssd_score + 0.35 * sdnn_score + 0.20 * pnn50_score

    parasympathetic_ratio = rmssd_ms / max(heart_rate_bpm, 1.0)
    autonomic_balance_index = clamp(50.0 + (parasympathetic_ratio - 0.45) * 120.0, 0.0, 100.0)
    if autonomic_balance_index < 40.0:
        autonomic_balance_state = "교감신경 우세"
    elif autonomic_balance_index <= 60.0:
        autonomic_balance_state = "균형"
    else:
        autonomic_balance_state = "부교감신경 우세"

    return {
        "hrv_score": float(hrv_score),
        "autonomic_balance_index": float(autonomic_balance_index),
        "autonomic_balance_state": autonomic_balance_state,
    }


def _apply_open_source_vital_adjustments(
    context: dict[str, Any],
    heart_rate_metrics: dict[str, Any],
    hrv_metrics: dict[str, Any],
    open_source_report: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not open_source_report.get("available"):
        return context, heart_rate_metrics, hrv_metrics, open_source_report

    adjusted_context = dict(context)
    adjusted_heart_rate_metrics = dict(heart_rate_metrics)
    adjusted_hrv_metrics = dict(hrv_metrics)
    applied_outputs = open_source_report.setdefault("applied_outputs", {})
    derived = open_source_report.get("derived") or {}
    neurokit_report = open_source_report.get("neurokit2") or {}

    oss_signal_quality = float(derived.get("signal_quality_score") or 0.0)
    if oss_signal_quality > 0.0:
        rule_based_quality = float(context["signal_quality_score"])
        adjusted_context["signal_quality_score_rule_based"] = rule_based_quality
        adjusted_context["signal_quality_score_oss"] = oss_signal_quality
        adjusted_context["signal_quality_score"] = float(
            0.55 * rule_based_quality + 0.45 * oss_signal_quality
        )
        applied_outputs["signal_quality_score"] = float(adjusted_context["signal_quality_score"])

    baseline_hr = float(adjusted_heart_rate_metrics["heart_rate_bpm"])
    neurokit_hr = float(neurokit_report.get("heart_rate_bpm") or 0.0)
    if 40.0 <= neurokit_hr <= 180.0:
        adjusted_heart_rate_metrics["heart_rate_rule_based_bpm"] = baseline_hr
        adjusted_heart_rate_metrics["heart_rate_oss_bpm"] = neurokit_hr
        if baseline_hr <= 0.0:
            adjusted_heart_rate_metrics["heart_rate_bpm"] = float(neurokit_hr)
        elif abs(neurokit_hr - baseline_hr) <= 12.0:
            adjusted_heart_rate_metrics["heart_rate_bpm"] = float(0.65 * baseline_hr + 0.35 * neurokit_hr)
        else:
            warnings.append("NeuroKit2 심박수와 기존 심박수 차이가 커서 규칙 기반 값을 유지합니다.")
        adjusted_heart_rate_metrics["ibi_mean_ms"] = float(60000.0 / max(float(adjusted_heart_rate_metrics["heart_rate_bpm"]), 1.0))
        applied_outputs["1.1"] = float(adjusted_heart_rate_metrics["heart_rate_bpm"])

    neurokit_sdnn = float(neurokit_report.get("sdnn_ms") or 0.0)
    neurokit_rmssd = float(neurokit_report.get("rmssd_ms") or 0.0)
    neurokit_pnn50 = float(neurokit_report.get("pnn50") or 0.0)
    if neurokit_sdnn > 0.0 or neurokit_rmssd > 0.0 or neurokit_pnn50 > 0.0:
        baseline_sdnn = float(adjusted_hrv_metrics["sdnn_ms"])
        baseline_rmssd = float(adjusted_hrv_metrics["rmssd_ms"])
        baseline_pnn50 = float(adjusted_hrv_metrics["pnn50"])
        adjusted_hrv_metrics["sdnn_ms_rule_based"] = baseline_sdnn
        adjusted_hrv_metrics["rmssd_ms_rule_based"] = baseline_rmssd
        adjusted_hrv_metrics["pnn50_rule_based"] = baseline_pnn50
        adjusted_hrv_metrics["sdnn_ms_oss"] = neurokit_sdnn
        adjusted_hrv_metrics["rmssd_ms_oss"] = neurokit_rmssd
        adjusted_hrv_metrics["pnn50_oss"] = neurokit_pnn50

        adjusted_hrv_metrics["sdnn_ms"] = float(neurokit_sdnn if baseline_sdnn <= 0.0 else 0.60 * baseline_sdnn + 0.40 * neurokit_sdnn)
        adjusted_hrv_metrics["rmssd_ms"] = float(neurokit_rmssd if baseline_rmssd <= 0.0 else 0.60 * baseline_rmssd + 0.40 * neurokit_rmssd)
        adjusted_hrv_metrics["pnn50"] = float(neurokit_pnn50 if baseline_pnn50 <= 0.0 else 0.60 * baseline_pnn50 + 0.40 * neurokit_pnn50)
        adjusted_hrv_metrics.update(
            _recalculate_hrv_support_metrics(
                float(adjusted_heart_rate_metrics["heart_rate_bpm"]),
                float(adjusted_hrv_metrics["sdnn_ms"]),
                float(adjusted_hrv_metrics["rmssd_ms"]),
                float(adjusted_hrv_metrics["pnn50"]),
            )
        )
        applied_outputs["1.2_sdnn"] = float(adjusted_hrv_metrics["sdnn_ms"])
        applied_outputs["1.2_rmssd"] = float(adjusted_hrv_metrics["rmssd_ms"])
        applied_outputs["1.2_pnn50"] = float(adjusted_hrv_metrics["pnn50"])

    return adjusted_context, adjusted_heart_rate_metrics, adjusted_hrv_metrics, open_source_report


def _apply_open_source_proxy_adjustments(
    circulation_metrics: dict[str, Any],
    vascular_health_metrics: dict[str, Any],
    open_source_report: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not open_source_report.get("available"):
        return circulation_metrics, vascular_health_metrics, open_source_report

    adjusted_circulation_metrics = dict(circulation_metrics)
    adjusted_vascular_health_metrics = dict(vascular_health_metrics)
    applied_outputs = open_source_report.setdefault("applied_outputs", {})
    derived = open_source_report.get("derived") or {}
    pyppg_report = open_source_report.get("pyppg") or {}
    quality_score = float(derived.get("signal_quality_score") or 0.0)

    circulation_proxy_score = float(derived.get("circulation_proxy_score") or 0.0)
    perfusion_index_proxy = float(derived.get("perfusion_index_proxy") or 0.0)
    if circulation_proxy_score > 0.0:
        baseline_circulation = float(circulation_metrics["circulation_score"])
        oss_weight = float(clamp(0.10 + 0.15 * (quality_score / 100.0), 0.10, 0.25))
        adjusted_circulation_metrics["circulation_score_rule_based"] = baseline_circulation
        adjusted_circulation_metrics["circulation_score_oss"] = circulation_proxy_score
        adjusted_circulation_metrics["circulation_score"] = float(
            (1.0 - oss_weight) * baseline_circulation + oss_weight * circulation_proxy_score
        )
        if perfusion_index_proxy > 0.0:
            baseline_perfusion_index = float(circulation_metrics.get("perfusion_index", 0.0) or 0.0)
            adjusted_circulation_metrics["perfusion_index_rule_based"] = baseline_perfusion_index
            adjusted_circulation_metrics["perfusion_index_oss"] = perfusion_index_proxy
            adjusted_circulation_metrics["perfusion_index"] = float(
                (1.0 - oss_weight) * baseline_perfusion_index + oss_weight * perfusion_index_proxy
            )
        adjusted_circulation_metrics["oss_weight"] = oss_weight
        applied_outputs["1.4"] = float(adjusted_circulation_metrics["circulation_score"])
    else:
        warnings.append("pyPPG 순환 프록시가 충분하지 않아 1.4는 기존 계산을 유지합니다.")

    vascular_proxy_score = float(derived.get("vascular_proxy_score") or 0.0)
    biomarkers = pyppg_report.get("biomarkers") or {}
    if vascular_proxy_score > 0.0:
        baseline_vascular_score = float(vascular_health_metrics["vascular_health_score"])
        oss_weight = float(clamp(0.08 + 0.12 * (quality_score / 100.0), 0.08, 0.20))
        adjusted_vascular_health_metrics["vascular_health_score_rule_based"] = baseline_vascular_score
        adjusted_vascular_health_metrics["vascular_health_score_oss"] = vascular_proxy_score
        adjusted_vascular_health_metrics["vascular_health_score"] = float(
            (1.0 - oss_weight) * baseline_vascular_score + oss_weight * vascular_proxy_score
        )
        adjusted_vascular_health_metrics["ai_oss"] = biomarkers.get("ai")
        adjusted_vascular_health_metrics["agi_oss"] = biomarkers.get("agi")
        adjusted_vascular_health_metrics["agimod_oss"] = biomarkers.get("agimod")
        adjusted_vascular_health_metrics["tsys_tdia_ratio_oss"] = biomarkers.get("tsys_tdia_ratio")
        adjusted_vascular_health_metrics["oss_weight"] = oss_weight
        applied_outputs["1.5"] = float(adjusted_vascular_health_metrics["vascular_health_score"])
    else:
        warnings.append("pyPPG 혈관 프록시가 충분하지 않아 1.5는 기존 계산을 유지합니다.")

    return adjusted_circulation_metrics, adjusted_vascular_health_metrics, open_source_report


def _build_output_quality_report(
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
    signal_quality_score = float(context["signal_quality_score"])
    duration_s = float(context["duration_s"])
    peak_count = int(context["peak_indices"].size)
    peak_count_score = clamp((peak_count - 4) / 12.0 * 100.0, 0.0, 100.0)
    duration_score = clamp(duration_s / 60.0 * 100.0, 0.0, 100.0)
    beat_source_penalty = 10.0 if context["beat_source"] != "ppg_peak_detection" else 0.0
    camera_quality_score = float(camera_report.get("camera_quality_score") or 0.0)
    camera_support_score = camera_quality_score if camera_report.get("available") else 0.0
    calibrated = bool(blood_pressure_metrics.get("calibrated"))

    outputs: dict[str, dict[str, Any]] = {}

    hr_confidence = clamp(0.65 * signal_quality_score + 0.20 * peak_count_score + 0.15 * camera_support_score - beat_source_penalty, 0.0, 100.0)
    outputs["1.1"] = {
        "label": "심박수",
        "confidence_score": float(hr_confidence),
        "available": bool(heart_rate_metrics["heart_rate_bpm"] > 0.0 and hr_confidence >= 35.0),
        "status_label": "사용 가능" if heart_rate_metrics["heart_rate_bpm"] > 0.0 and hr_confidence >= 35.0 else "무응답 권장",
        "reason": "신호 품질과 박동 수를 기준으로 평가했습니다.",
    }

    hrv_confidence = clamp(0.55 * signal_quality_score + 0.25 * duration_score + 0.20 * peak_count_score - beat_source_penalty, 0.0, 100.0)
    hrv_available = peak_count >= 8 and duration_s >= 25.0 and hrv_confidence >= 35.0
    outputs["1.2"] = {
        "label": "HRV",
        "confidence_score": float(hrv_confidence),
        "available": bool(hrv_available),
        "status_label": "사용 가능" if hrv_available else "무응답 권장",
        "reason": "신호 품질, 측정 시간, 박동 수를 기준으로 평가했습니다.",
    }

    stress_confidence = clamp(0.45 * hr_confidence + 0.45 * hrv_confidence + 0.10 * camera_support_score, 0.0, 100.0)
    stress_available = outputs["1.1"]["available"] and outputs["1.2"]["available"] and stress_confidence >= 40.0
    outputs["1.3"] = {
        "label": "스트레스",
        "confidence_score": float(stress_confidence),
        "available": bool(stress_available),
        "status_label": "사용 가능" if stress_available else "무응답 권장",
        "reason": "심박수와 HRV 신뢰도를 함께 반영했습니다.",
    }

    circulation_camera_score = float(circulation_metrics.get("circulation_score_camera") or 0.0)
    circulation_confidence = clamp(
        0.60 * signal_quality_score
        + 0.15 * peak_count_score
        + 0.15 * camera_quality_score
        + 0.10 * circulation_camera_score,
        0.0,
        100.0,
    )
    circulation_available = circulation_confidence >= 35.0
    outputs["1.4"] = {
        "label": "순환",
        "confidence_score": float(circulation_confidence),
        "available": bool(circulation_available),
        "status_label": "사용 가능" if circulation_available else "무응답 권장",
        "reason": "PPG 품질과 카메라 관류 프록시를 함께 반영했습니다.",
    }

    vascular_camera_score = float(vascular_health_metrics.get("vascular_health_score_camera") or 0.0)
    vascular_confidence = clamp(
        0.58 * signal_quality_score
        + 0.14 * peak_count_score
        + 0.14 * camera_quality_score
        + 0.14 * vascular_camera_score,
        0.0,
        100.0,
    )
    vascular_available = vascular_health_metrics.get("reflection_index") is not None and vascular_confidence >= 35.0
    outputs["1.5"] = {
        "label": "혈관 건강",
        "confidence_score": float(vascular_confidence),
        "available": bool(vascular_available),
        "status_label": "사용 가능" if vascular_available else "무응답 권장",
        "reason": "파형 형태 안정성과 카메라 혈관 프록시를 반영했습니다.",
    }

    age_penalty = 0.0 if user_profile.age is not None else 15.0
    vascular_age_confidence = clamp(0.45 * vascular_confidence + 0.35 * hrv_confidence + 0.20 * hr_confidence - age_penalty, 0.0, 100.0)
    vascular_age_available = outputs["1.5"]["available"] and outputs["1.2"]["available"] and vascular_age_confidence >= 35.0
    outputs["1.6"] = {
        "label": "혈관 나이",
        "confidence_score": float(vascular_age_confidence),
        "available": bool(vascular_age_available),
        "status_label": "사용 가능" if vascular_age_available else "무응답 권장",
        "reason": "혈관 건강, HRV, 입력된 나이 정보를 반영했습니다.",
    }

    calibration_score = 100.0 if calibrated else 25.0
    blood_pressure_confidence = clamp(
        0.30 * circulation_confidence
        + 0.30 * vascular_confidence
        + 0.15 * hr_confidence
        + 0.10 * camera_quality_score
        + 0.15 * calibration_score,
        0.0,
        100.0,
    )
    blood_pressure_available = outputs["1.4"]["available"] and outputs["1.5"]["available"] and blood_pressure_confidence >= 30.0
    outputs["1.7"] = {
        "label": "혈압",
        "confidence_score": float(blood_pressure_confidence),
        "available": bool(blood_pressure_available),
        "status_label": "사용 가능" if blood_pressure_available else "무응답 권장",
        "reason": "순환, 혈관 건강, 보정 여부를 반영했습니다.",
        "trend_only": not calibrated,
    }

    no_read_outputs = [f"{key} {value['label']}" for key, value in outputs.items() if not value["available"]]
    overall_confidence_score = float(np.mean([float(item["confidence_score"]) for item in outputs.values()]))

    return {
        "overall_confidence_score": overall_confidence_score,
        "no_read_outputs": no_read_outputs,
        "outputs": outputs,
    }


def _format_output_confidence_compact(quality_report: dict[str, Any]) -> str:
    outputs = quality_report.get("outputs") or {}
    ordered_keys = ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"]
    parts: list[str] = []
    for key in ordered_keys:
        item = outputs.get(key)
        if not item:
            continue
        parts.append(f"{key} {float(item['confidence_score']):.0f}")
    return " | ".join(parts)


def _stress_state_from_score(score: float) -> str:
    if score < 33.0:
        return translate_stress_state("stable")
    if score < 66.0:
        return translate_stress_state("normal")
    return translate_stress_state("tense")


def _blood_pressure_trend_from_values(
    user_profile: UserProfile,
    estimated_sbp: float,
    estimated_dbp: float,
) -> str:
    base_sbp = float(user_profile.calibration_sbp) if user_profile.calibration_sbp is not None else 120.0
    base_dbp = float(user_profile.calibration_dbp) if user_profile.calibration_dbp is not None else 80.0
    delta_sbp = estimated_sbp - base_sbp
    delta_dbp = estimated_dbp - base_dbp
    if delta_sbp > 5.0 or delta_dbp > 4.0:
        return translate_bp_trend("rising")
    if delta_sbp < -5.0 or delta_dbp < -4.0:
        return translate_bp_trend("falling")
    return translate_bp_trend("stable")


def _apply_ml_inference(
    *,
    context: dict[str, Any],
    user_profile: UserProfile,
    camera_report: dict[str, Any],
    heart_rate_metrics: dict[str, Any],
    hrv_metrics: dict[str, Any],
    stress_metrics: dict[str, Any],
    circulation_metrics: dict[str, Any],
    vascular_health_metrics: dict[str, Any],
    vascular_age_metrics: dict[str, Any],
    blood_pressure_metrics: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    adjusted_stress_metrics = dict(stress_metrics)
    adjusted_circulation_metrics = dict(circulation_metrics)
    adjusted_vascular_health_metrics = dict(vascular_health_metrics)
    adjusted_vascular_age_metrics = dict(vascular_age_metrics)
    adjusted_blood_pressure_metrics = dict(blood_pressure_metrics)

    feature_payload = build_feature_payload(
        context=context,
        user_profile=user_profile,
        camera_report=camera_report,
        heart_rate_metrics=heart_rate_metrics,
        hrv_metrics=hrv_metrics,
        stress_metrics=stress_metrics,
        circulation_metrics=circulation_metrics,
        vascular_health_metrics=vascular_health_metrics,
        vascular_age_metrics=vascular_age_metrics,
        blood_pressure_metrics=blood_pressure_metrics,
    )
    inference = run_ml_inference(feature_payload)
    outputs = inference.outputs

    baseline_stress = float(stress_metrics["stress_score"])
    baseline_circulation = float(circulation_metrics["circulation_score"])
    baseline_vascular_health = float(vascular_health_metrics["vascular_health_score"])
    baseline_vascular_age = float(vascular_age_metrics["vascular_age_estimate"])
    baseline_sbp = float(blood_pressure_metrics["estimated_sbp"])
    baseline_dbp = float(blood_pressure_metrics["estimated_dbp"])
    chronological_age = int(adjusted_vascular_age_metrics.get("chronological_age_reference") or (user_profile.age or 45))

    ml_stress_score = float(outputs["stress_score"])
    ml_circulation_score = float(outputs["circulation_score"])
    ml_vascular_health_score = float(outputs["vascular_health_score"])
    ml_vascular_age = float(outputs["vascular_age_estimate"])
    ml_sbp = float(outputs["estimated_sbp"])
    ml_dbp = float(outputs["estimated_dbp"])

    adjusted_stress_metrics["stress_score_rule_based"] = baseline_stress
    adjusted_stress_metrics["stress_score_ml"] = ml_stress_score
    adjusted_stress_metrics["stress_score"] = ml_stress_score
    adjusted_stress_metrics["stress_state"] = _stress_state_from_score(ml_stress_score)

    adjusted_circulation_metrics["circulation_score_rule_based"] = baseline_circulation
    adjusted_circulation_metrics["circulation_score_ml"] = ml_circulation_score
    adjusted_circulation_metrics["circulation_score"] = ml_circulation_score

    adjusted_vascular_health_metrics["vascular_health_score_rule_based"] = baseline_vascular_health
    adjusted_vascular_health_metrics["vascular_health_score_ml"] = ml_vascular_health_score
    adjusted_vascular_health_metrics["vascular_health_score"] = ml_vascular_health_score

    adjusted_vascular_age_metrics["vascular_age_estimate_rule_based"] = baseline_vascular_age
    adjusted_vascular_age_metrics["vascular_age_estimate_ml"] = ml_vascular_age
    adjusted_vascular_age_metrics["vascular_age_estimate"] = ml_vascular_age
    adjusted_vascular_age_metrics["vascular_age_gap"] = float(ml_vascular_age - chronological_age)

    adjusted_blood_pressure_metrics["estimated_sbp_rule_based"] = baseline_sbp
    adjusted_blood_pressure_metrics["estimated_dbp_rule_based"] = baseline_dbp
    adjusted_blood_pressure_metrics["estimated_sbp_ml"] = ml_sbp
    adjusted_blood_pressure_metrics["estimated_dbp_ml"] = ml_dbp
    adjusted_blood_pressure_metrics["estimated_sbp"] = ml_sbp
    adjusted_blood_pressure_metrics["estimated_dbp"] = ml_dbp
    adjusted_blood_pressure_metrics["blood_pressure_trend"] = _blood_pressure_trend_from_values(
        user_profile,
        ml_sbp,
        ml_dbp,
    )

    inference_report = {
        "available": True,
        "bundle_path": inference.bundle_path,
        "bundle_version": inference.bundle_version,
        "bootstrap_bundle": inference.bootstrap_bundle,
        "model_family": "scikit-learn",
        "applied_outputs": {
            "1.3": ml_stress_score,
            "1.4": ml_circulation_score,
            "1.5": ml_vascular_health_score,
            "1.6": ml_vascular_age,
            "1.7_sbp": ml_sbp,
            "1.7_dbp": ml_dbp,
        },
    }

    if inference.bootstrap_bundle:
        warnings.append("ML 추론은 현재 기본 번들로 동작 중이며, 실측 학습 데이터로 교체하면 결과 신뢰도를 더 높일 수 있습니다.")

    return (
        adjusted_stress_metrics,
        adjusted_circulation_metrics,
        adjusted_vascular_health_metrics,
        adjusted_vascular_age_metrics,
        adjusted_blood_pressure_metrics,
        inference_report,
    )


def prepare_context(dataset: SignalDataset) -> dict[str, Any]:
    sample_rate_hz = dataset.sample_rate_hz or estimate_sample_rate(dataset.timestamps_s)
    if sample_rate_hz <= 0.0:
        raise ValueError("데이터셋에서 샘플링 주파수를 계산하지 못했습니다.")

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
        raise ValueError("1.1 ~ 1.7 지표를 계산하기에 충분한 박동을 검출하지 못했습니다.")

    onset_indices = find_onsets(filtered_ppg, peak_indices, sample_rate_hz)
    pulse_features = extract_pulse_features(filtered_ppg, peak_indices, onset_indices, sample_rate_hz)
    average_pulse = build_average_pulse(filtered_ppg, onset_indices)

    duration_s = float(dataset.timestamps_s[-1] - dataset.timestamps_s[0]) if dataset.timestamps_s.size >= 2 else 0.0
    signal_quality_score = calculate_signal_quality(raw_ppg, filtered_ppg)

    return {
        "sample_rate_hz": sample_rate_hz,
        "warnings": warnings,
        "raw_ppg": raw_ppg,
        "filtered_ppg": filtered_ppg,
        "peak_indices": peak_indices,
        "onset_indices": onset_indices,
        "pulse_features": pulse_features,
        "average_pulse": average_pulse,
        "beat_source": beat_source,
        "duration_s": duration_s,
        "signal_quality_score": signal_quality_score,
    }


def run_stepwise_analysis(
    dataset: SignalDataset,
    user_profile: UserProfile,
    camera_summary: dict[str, Any] | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    context = prepare_context(dataset)
    warnings = list(context["warnings"])
    open_source_report = analyze_contact_ppg_with_oss(context["raw_ppg"], context["sample_rate_hz"])
    for item in open_source_report.get("warnings") or []:
        warnings.append(f"오픈소스 분석 참고: {item}")
    for item in open_source_report.get("errors") or []:
        warnings.append(f"오픈소스 분석 경고: {item}")
    ml_report: dict[str, Any] = {
        "available": False,
        "bundle_path": None,
        "bundle_version": None,
        "bootstrap_bundle": False,
        "model_family": None,
        "applied_outputs": {},
    }

    def notify(index: int, label: str) -> None:
        if progress is not None:
            progress(index, translate_step_label(label))

    notify(1, "1.1 Heart rate")
    heart_rate_metrics = calculate_hr_metrics(context["peak_indices"], context["sample_rate_hz"])

    notify(2, "1.2 HRV")
    hrv_metrics = calculate_hrv_metrics(context["peak_indices"], context["sample_rate_hz"])
    context, heart_rate_metrics, hrv_metrics, open_source_report = _apply_open_source_vital_adjustments(
        context,
        heart_rate_metrics,
        hrv_metrics,
        open_source_report,
        warnings,
    )
    ippg_heart_rate_bpm = float(heart_rate_metrics["heart_rate_bpm"])
    camera_report, fused_heart_rate_bpm = _build_camera_report(
        camera_summary,
        float(context["signal_quality_score"]),
        ippg_heart_rate_bpm,
        warnings,
    )
    if camera_report["fusion_applied"]:
        heart_rate_metrics["heart_rate_bpm"] = fused_heart_rate_bpm
        hrv_metrics.update(
            _recalculate_hrv_support_metrics(
                float(heart_rate_metrics["heart_rate_bpm"]),
                float(hrv_metrics["sdnn_ms"]),
                float(hrv_metrics["rmssd_ms"]),
                float(hrv_metrics["pnn50"]),
            )
        )

    notify(3, "1.3 Stress")
    stress_metrics = calculate_stress_metrics(float(heart_rate_metrics["heart_rate_bpm"]), hrv_metrics)

    notify(4, "1.4 Circulation")
    circulation_metrics = calculate_circulation_metrics(
        context["pulse_features"],
        context["filtered_ppg"],
        dataset.aux,
    )

    notify(5, "1.5 Vascular health")
    vascular_health_metrics = calculate_vascular_health_metrics(
        context["average_pulse"],
        context["pulse_features"],
    )
    camera_report, circulation_metrics, vascular_health_metrics = _apply_camera_proxy_adjustments(
        camera_report,
        circulation_metrics,
        vascular_health_metrics,
        warnings,
    )
    circulation_metrics, vascular_health_metrics, open_source_report = _apply_open_source_proxy_adjustments(
        circulation_metrics,
        vascular_health_metrics,
        open_source_report,
        warnings,
    )

    notify(6, "1.6 Vascular age")
    vascular_age_metrics = calculate_vascular_age_metrics(
        user_profile,
        float(heart_rate_metrics["heart_rate_bpm"]),
        hrv_metrics,
        circulation_metrics,
        vascular_health_metrics,
        context["pulse_features"],
    )

    notify(7, "1.7 Blood pressure")
    blood_pressure_metrics = calculate_blood_pressure_metrics(
        user_profile,
        float(heart_rate_metrics["heart_rate_bpm"]),
        circulation_metrics,
        vascular_health_metrics,
        vascular_age_metrics,
        context["pulse_features"],
    )
    (
        stress_metrics,
        circulation_metrics,
        vascular_health_metrics,
        vascular_age_metrics,
        blood_pressure_metrics,
        ml_report,
    ) = _apply_ml_inference(
        context=context,
        user_profile=user_profile,
        camera_report=camera_report,
        heart_rate_metrics=heart_rate_metrics,
        hrv_metrics=hrv_metrics,
        stress_metrics=stress_metrics,
        circulation_metrics=circulation_metrics,
        vascular_health_metrics=vascular_health_metrics,
        vascular_age_metrics=vascular_age_metrics,
        blood_pressure_metrics=blood_pressure_metrics,
        warnings=warnings,
    )

    if not circulation_metrics["aux_channel_available"]:
        warnings.append("현재 단일 PPG 하드웨어에서는 좌우 채널 차이 항목을 계산할 수 없습니다.")
    if not blood_pressure_metrics["calibrated"]:
        warnings.append("개인 커프 보정값이 없어 혈압 결과는 추세 중심 참고값입니다.")
    if user_profile.age is None:
        warnings.append("나이가 입력되지 않아 혈관 나이는 기본 기준 나이 45세를 사용했습니다.")
    if context["signal_quality_score"] < 35.0:
        warnings.append("신호 품질이 낮습니다. 센서를 다시 밀착하고 손가락 움직임을 줄여주세요.")
    quality_report = _build_output_quality_report(
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

    return {
        "metadata": {
            "sample_rate_hz": float(context["sample_rate_hz"]),
            "sample_count": int(context["raw_ppg"].size),
            "duration_s": float(context["duration_s"]),
            "signal_quality_score": float(context["signal_quality_score"]),
            "beat_detection_source": context["beat_source"],
            "measurement_mode": camera_report["measurement_mode"],
            "measurement_mode_label": camera_report["measurement_mode_label"],
            "open_source_analysis_applied": bool(open_source_report.get("available")),
            "ml_inference_applied": bool(ml_report.get("available")),
        },
        "camera": camera_report,
        "open_source": open_source_report,
        "ml": ml_report,
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


def format_console_summary(report: dict[str, Any]) -> str:
    heart_rate = report["heart_rate"]
    hrv = report["hrv"]
    stress = report["stress"]
    circulation = report["circulation"]
    vascular_health = report["vascular_health"]
    vascular_age = report["vascular_age"]
    blood_pressure = report["blood_pressure"]
    metadata = report["metadata"]
    camera = report.get("camera") or {}
    open_source_report = report.get("open_source") or {}
    ml_report = report.get("ml") or {}
    quality = report.get("quality") or {}
    no_read_outputs = quality.get("no_read_outputs") or []

    lines = [
        "",
        "========================================",
        "1.1 ~ 1.7 통합 분석 결과",
        "========================================",
        f"측정 모드      : {metadata.get('measurement_mode_label', 'iPPG 단독')}",
        f"신호 품질      : {metadata['signal_quality_score']:.2f} / 100",
        f"전체 신뢰도    : {float(quality.get('overall_confidence_score') or 0.0):.2f} / 100",
        f"ML 추론        : {'적용' if ml_report.get('available') else '미적용'}",
        f"1.1 심박수     : {heart_rate['heart_rate_bpm']:.2f} bpm",
        f"1.2 HRV        : SDNN {hrv['sdnn_ms']:.2f} ms | RMSSD {hrv['rmssd_ms']:.2f} ms | pNN50 {hrv['pnn50']:.2f}% | 점수 {hrv['hrv_score']:.2f}",
        f"1.3 스트레스   : {stress['stress_score']:.2f} / 100 | {translate_stress_state(str(stress['stress_state']))}",
        f"1.4 순환       : {circulation['circulation_score']:.2f} / 100",
        f"1.5 혈관 건강  : {vascular_health['vascular_health_score']:.2f} / 100 | 반사 지수 {vascular_health['reflection_index']}",
        f"1.6 혈관 나이  : {vascular_age['vascular_age_estimate']:.1f}세 | 차이 {vascular_age['vascular_age_gap']:+.1f}",
        f"1.7 혈압       : {blood_pressure['estimated_sbp']:.1f}/{blood_pressure['estimated_dbp']:.1f} mmHg | {translate_bp_trend(str(blood_pressure['blood_pressure_trend']))}",
        f"출력 신뢰도    : {_format_output_confidence_compact(quality)}",
        f"무응답 권장    : {', '.join(no_read_outputs) if no_read_outputs else '없음'}",
    ]

    if open_source_report.get("available"):
        engine_label = str(open_source_report.get("engine_label") or "-")
        derived = open_source_report.get("derived") or {}
        oss_quality_text = f"{float(derived.get('signal_quality_score') or 0.0):.1f}"
        nk_report = open_source_report.get("neurokit2") or {}
        pyppg_report = open_source_report.get("pyppg") or {}
        nk_hr_text = (
            f"{float(nk_report.get('heart_rate_bpm') or 0.0):.2f} bpm"
            if nk_report.get("heart_rate_bpm") is not None
            else "-"
        )
        pyppg_sqi_text = (
            f"{float(pyppg_report.get('sqi_score') or 0.0):.1f}"
            if pyppg_report.get("sqi_score") is not None
            else "-"
        )
        lines.append(
            f"오픈소스 분석  : {engine_label} | OSS 품질 {oss_quality_text} | NeuroKit2 HR {nk_hr_text} | pyPPG SQI {pyppg_sqi_text}"
        )

    if camera.get("available"):
        camera_hr_text = f"{float(camera['camera_hr_bpm']):.2f} bpm" if camera.get("camera_hr_bpm") is not None else "미검출"
        face_text = f"{float(camera['face_detection_ratio']) * 100.0:.1f}%" if camera.get("face_detection_ratio") is not None else "-"
        signal_text = str(camera.get("selected_signal_label") or "-")
        perfusion_text = f"{float(camera.get('camera_perfusion_proxy_score') or 0.0):.1f}"
        vascular_text = f"{float(camera.get('camera_vascular_proxy_score') or 0.0):.1f}"
        lines.append(
            f"카메라 보조    : {camera.get('measurement_mode_label', '카메라 보조 분석')} | 카메라 HR {camera_hr_text} | 얼굴 검출률 {face_text} | 신호 {signal_text} | 관류 프록시 {perfusion_text} | 혈관 프록시 {vascular_text}"
        )
    if ml_report.get("available"):
        bundle_label = str(ml_report.get("bundle_version") or "-")
        bundle_kind = "기본 번들" if ml_report.get("bootstrap_bundle") else "사용자 학습 번들"
        lines.append(f"ML 모델        : {bundle_label} | {bundle_kind}")

    if report["warnings"]:
        lines.append("경고:")
        lines.extend(f"- {warning}" for warning in report["warnings"])

    return "\n".join(lines)


def print_progress(step_index: int, label: str) -> None:
    print(f"[{step_index}/7] {label} 완료")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture once, calculate section 1.1 to 1.7 in sequence, and print one final combined result block."
    )
    parser.add_argument("--port", help="Serial port such as COM3. When omitted, --csv-input must be used.")
    parser.add_argument("--baud", type=int, default=1000000, help="Serial baud rate. Default: 1000000")
    parser.add_argument("--duration", type=float, default=60.0, help="Measurement duration in seconds. Default: 60")
    parser.add_argument("--csv-input", help="Analyze an existing capture CSV instead of reading from serial.")
    parser.add_argument("--sample-rate", type=float, default=200.0, help="Fallback sample rate. Default: 200 Hz")
    parser.add_argument("--age", type=int, help="Chronological age used for vascular age estimation.")
    parser.add_argument("--sex", default="unknown", help="male, female, or unknown")
    parser.add_argument("--calibration-sbp", type=float, help="Personal cuff-calibrated systolic baseline.")
    parser.add_argument("--calibration-dbp", type=float, help="Personal cuff-calibrated diastolic baseline.")
    parser.add_argument("--camera-summary", help="Optional camera_rppg_summary.json path for camera+iPPG fusion analysis.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="Directory for captured CSV and report files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.port and not args.csv_input:
        raise SystemExit("--port로 라이브 측정을 하거나 --csv-input으로 오프라인 분석 파일을 지정하세요.")

    output_dir = Path(args.output_dir).resolve()
    capture_path: Path | None = None
    camera_summary: dict[str, Any] | None = load_camera_summary(Path(args.camera_summary).resolve()) if args.camera_summary else None

    if args.csv_input:
        print("기존 CSV 입력 파일을 사용해 순차 분석을 진행합니다.")
        csv_path = Path(args.csv_input).resolve()
        dataset = load_dataset_from_csv(csv_path, fallback_sample_rate_hz=args.sample_rate)
        if camera_summary is None:
            auto_camera_summary_path = csv_path.parent / "camera_rppg_summary.json"
            camera_summary = load_camera_summary(auto_camera_summary_path)
            if camera_summary is not None:
                print(f"카메라 요약을 자동으로 불러왔습니다: {auto_camera_summary_path}")
    else:
        print("0/7 단계: 원시 PPG 데이터를 수집합니다. 측정이 끝날 때까지 손가락을 안정적으로 유지하세요.")
        samples = capture_serial_session(args.port, args.baud, args.duration, args.sample_rate)
        capture_path = output_dir / "capture.csv"
        write_capture_csv(capture_path, samples)
        dataset = load_dataset_from_csv(capture_path, fallback_sample_rate_hz=args.sample_rate)
        print("신호 수집이 끝났습니다. 1.1 ~ 1.7 계산을 시작합니다.")

    report = run_stepwise_analysis(dataset, build_user_profile(args), camera_summary=camera_summary, progress=print_progress)
    report_path, summary_path = write_report_files(output_dir, report, capture_path=capture_path)

    print(format_console_summary(report))
    print("")
    print(f"리포트 JSON : {report_path}")
    print(f"요약 TXT    : {summary_path}")
    if capture_path is not None:
        print(f"캡처 CSV    : {capture_path}")


if __name__ == "__main__":
    main()
