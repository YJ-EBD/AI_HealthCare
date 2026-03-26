from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diagnostics import APP_DATA_DIR, log_event


MODELS_DIR = APP_DATA_DIR / "models"
BUNDLE_PATH = MODELS_DIR / "bootstrap_ml_bundle.joblib"

FEATURE_ORDER = [
    "signal_quality_score",
    "duration_s",
    "sample_rate_hz",
    "heart_rate_bpm",
    "ibi_mean_ms",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "hrv_score",
    "stress_score_baseline",
    "circulation_score_baseline",
    "vascular_health_score_baseline",
    "vascular_age_estimate_baseline",
    "vascular_age_gap_baseline",
    "estimated_sbp_baseline",
    "estimated_dbp_baseline",
    "reflection_index",
    "pulse_shape_score",
    "perfusion_index",
    "median_rise_time_s",
    "median_pulse_area",
    "camera_available",
    "camera_quality_score",
    "camera_hr_bpm",
    "camera_hr_difference_bpm",
    "camera_perfusion_proxy_score",
    "camera_vascular_proxy_score",
    "camera_roi_stability_score",
    "camera_rhythm_stability_score",
    "camera_face_detection_ratio",
    "camera_frame_count",
    "camera_fusion_applied",
    "camera_circulation_fusion_applied",
    "camera_vascular_fusion_applied",
    "calibrated",
    "age",
    "sex_male",
    "sex_female",
]


@dataclass(slots=True)
class InferenceResult:
    available: bool
    bundle_path: str
    bundle_version: str
    bootstrap_bundle: bool
    outputs: dict[str, float]


def _ensure_model_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _build_bootstrap_dataset(sample_count: int = 3000, seed: int = 42) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)

    signal_quality = rng.uniform(18.0, 95.0, sample_count)
    duration_s = rng.uniform(20.0, 120.0, sample_count)
    sample_rate_hz = rng.uniform(90.0, 240.0, sample_count)
    heart_rate_bpm = rng.uniform(48.0, 110.0, sample_count)
    ibi_mean_ms = 60000.0 / np.maximum(heart_rate_bpm, 1.0)
    sdnn_ms = rng.uniform(8.0, 80.0, sample_count)
    rmssd_ms = rng.uniform(6.0, 90.0, sample_count)
    pnn50 = rng.uniform(0.0, 40.0, sample_count)
    hrv_score = np.clip(0.45 * ((rmssd_ms - 10.0) / 50.0 * 100.0) + 0.35 * ((sdnn_ms - 15.0) / 55.0 * 100.0) + 0.20 * (pnn50 / 25.0 * 100.0), 0.0, 100.0)

    stress_baseline = np.clip(
        0.35 * ((heart_rate_bpm - 62.0) / 28.0 * 100.0)
        + 0.35 * ((40.0 - rmssd_ms) / 30.0 * 100.0)
        + 0.15 * ((45.0 - sdnn_ms) / 30.0 * 100.0)
        + 0.15 * ((15.0 - pnn50) / 15.0 * 100.0),
        0.0,
        100.0,
    )
    circulation_baseline = np.clip(signal_quality * 0.48 + rng.uniform(10.0, 35.0, sample_count), 0.0, 100.0)
    vascular_health_baseline = np.clip(signal_quality * 0.40 + hrv_score * 0.25 + rng.uniform(15.0, 35.0, sample_count), 0.0, 100.0)
    vascular_age_estimate_baseline = np.clip(rng.uniform(20.0, 70.0, sample_count) + 0.06 * (heart_rate_bpm - 65.0) + 0.05 * (45.0 - sdnn_ms), 18.0, 90.0)
    age = np.clip(rng.normal(46.0, 13.0, sample_count), 18.0, 85.0)
    vascular_age_gap_baseline = vascular_age_estimate_baseline - age
    reflection_index = np.clip(rng.normal(0.42, 0.08, sample_count), 0.18, 0.85)
    pulse_shape_score = np.clip(vascular_health_baseline * 0.75 + rng.uniform(-10.0, 10.0, sample_count), 0.0, 100.0)
    perfusion_index = np.clip(rng.normal(0.42, 0.12, sample_count), 0.05, 0.95)
    median_rise_time_s = np.clip(rng.normal(0.23, 0.05, sample_count), 0.08, 0.55)
    median_pulse_area = np.clip(rng.normal(95.0, 28.0, sample_count), 5.0, 260.0)

    camera_available = rng.integers(0, 2, sample_count).astype(float)
    camera_quality = camera_available * np.clip(signal_quality * 0.80 + rng.uniform(-12.0, 12.0, sample_count), 0.0, 100.0)
    camera_hr_bpm = np.where(
        camera_available > 0.5,
        np.clip(heart_rate_bpm + rng.normal(0.0, 4.0, sample_count), 40.0, 180.0),
        0.0,
    )
    camera_hr_difference = np.where(camera_available > 0.5, np.abs(camera_hr_bpm - heart_rate_bpm), 0.0)
    camera_perfusion_proxy = camera_available * np.clip(circulation_baseline * 0.85 + rng.uniform(-10.0, 10.0, sample_count), 0.0, 100.0)
    camera_vascular_proxy = camera_available * np.clip(vascular_health_baseline * 0.80 + rng.uniform(-12.0, 12.0, sample_count), 0.0, 100.0)
    camera_roi_stability = camera_available * np.clip(rng.normal(78.0, 18.0, sample_count), 0.0, 100.0)
    camera_rhythm_stability = camera_available * np.clip(rng.normal(75.0, 16.0, sample_count), 0.0, 100.0)
    camera_face_detection = camera_available * np.clip(rng.normal(0.60, 0.25, sample_count), 0.0, 1.0)
    camera_frame_count = camera_available * rng.integers(120, 900, sample_count)
    camera_fusion_applied = (camera_available > 0.5).astype(float)
    camera_circulation_fusion = ((camera_available > 0.5) & (camera_perfusion_proxy > 35.0)).astype(float)
    camera_vascular_fusion = ((camera_available > 0.5) & (camera_vascular_proxy > 35.0)).astype(float)

    calibrated = rng.integers(0, 2, sample_count).astype(float)
    sex_male = rng.integers(0, 2, sample_count).astype(float)
    sex_female = 1.0 - sex_male

    estimated_sbp_baseline = np.clip(
        118.0
        + 0.10 * (heart_rate_bpm - 70.0)
        + 0.10 * vascular_age_gap_baseline
        + 0.18 * (vascular_health_baseline - 60.0)
        + rng.normal(0.0, 4.0, sample_count),
        85.0,
        180.0,
    )
    estimated_dbp_baseline = np.clip(
        78.0
        + 0.07 * (heart_rate_bpm - 70.0)
        + 0.07 * vascular_age_gap_baseline
        + 0.12 * (vascular_health_baseline - 60.0)
        + rng.normal(0.0, 3.0, sample_count),
        50.0,
        120.0,
    )

    features = np.column_stack(
        [
            signal_quality,
            duration_s,
            sample_rate_hz,
            heart_rate_bpm,
            ibi_mean_ms,
            sdnn_ms,
            rmssd_ms,
            pnn50,
            hrv_score,
            stress_baseline,
            circulation_baseline,
            vascular_health_baseline,
            vascular_age_estimate_baseline,
            vascular_age_gap_baseline,
            estimated_sbp_baseline,
            estimated_dbp_baseline,
            reflection_index,
            pulse_shape_score,
            perfusion_index,
            median_rise_time_s,
            median_pulse_area,
            camera_available,
            camera_quality,
            camera_hr_bpm,
            camera_hr_difference,
            camera_perfusion_proxy,
            camera_vascular_proxy,
            camera_roi_stability,
            camera_rhythm_stability,
            camera_face_detection,
            camera_frame_count,
            camera_fusion_applied,
            camera_circulation_fusion,
            camera_vascular_fusion,
            calibrated,
            age,
            sex_male,
            sex_female,
        ]
    )

    outputs = {
        "stress_score": np.clip(
            0.55 * stress_baseline
            + 0.15 * camera_quality
            + 0.15 * (100.0 - hrv_score)
            + 0.15 * rng.normal(50.0, 10.0, sample_count),
            0.0,
            100.0,
        ),
        "circulation_score": np.clip(0.75 * circulation_baseline + 0.25 * camera_perfusion_proxy, 0.0, 100.0),
        "vascular_health_score": np.clip(0.78 * vascular_health_baseline + 0.22 * camera_vascular_proxy, 0.0, 100.0),
        "vascular_age_estimate": np.clip(age + 0.12 * (heart_rate_bpm - 68.0) + 0.08 * (50.0 - hrv_score) + 0.05 * (vascular_health_baseline - 60.0), 18.0, 90.0),
        "estimated_sbp": np.clip(estimated_sbp_baseline + 0.10 * camera_vascular_proxy + 0.04 * camera_quality, 85.0, 180.0),
        "estimated_dbp": np.clip(estimated_dbp_baseline + 0.06 * camera_vascular_proxy + 0.03 * camera_quality, 50.0, 120.0),
    }
    return features, outputs


def _make_bundle() -> dict[str, Any]:
    features, outputs = _build_bootstrap_dataset()

    stress_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=450, random_state=42)),
        ]
    )
    vascular_age_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=450, random_state=7)),
        ]
    )

    circulation_model = GradientBoostingRegressor(random_state=42)
    vascular_health_model = GradientBoostingRegressor(random_state=7)
    sbp_model = RandomForestRegressor(n_estimators=180, random_state=11, n_jobs=-1)
    dbp_model = RandomForestRegressor(n_estimators=180, random_state=13, n_jobs=-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        stress_model.fit(features, outputs["stress_score"])
        circulation_model.fit(features, outputs["circulation_score"])
        vascular_health_model.fit(features, outputs["vascular_health_score"])
        vascular_age_model.fit(features, outputs["vascular_age_estimate"])
        sbp_model.fit(features, outputs["estimated_sbp"])
        dbp_model.fit(features, outputs["estimated_dbp"])

    return {
        "bundle_version": "bootstrap_sklearn_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "bootstrap_bundle": True,
        "feature_order": FEATURE_ORDER,
        "models": {
            "stress_score": stress_model,
            "circulation_score": circulation_model,
            "vascular_health_score": vascular_health_model,
            "vascular_age_estimate": vascular_age_model,
            "estimated_sbp": sbp_model,
            "estimated_dbp": dbp_model,
        },
    }


def load_or_create_bundle(bundle_path: Path | None = None) -> dict[str, Any]:
    _ensure_model_dir()
    target_path = Path(bundle_path or BUNDLE_PATH)
    if target_path.exists():
        bundle = joblib.load(target_path)
        return bundle

    bundle = _make_bundle()
    joblib.dump(bundle, target_path)
    log_event("ml_inference", "부트스트랩 ML 번들을 생성했습니다.", details={"bundle_path": str(target_path)})
    return bundle


def build_feature_payload(
    *,
    context: dict[str, Any],
    user_profile: Any,
    camera_report: dict[str, Any],
    heart_rate_metrics: dict[str, Any],
    hrv_metrics: dict[str, Any],
    stress_metrics: dict[str, Any],
    circulation_metrics: dict[str, Any],
    vascular_health_metrics: dict[str, Any],
    vascular_age_metrics: dict[str, Any],
    blood_pressure_metrics: dict[str, Any],
) -> dict[str, float]:
    sex_text = str(getattr(user_profile, "sex", "unknown") or "unknown").lower()
    age_value = float(getattr(user_profile, "age", 45) or 45)
    payload = {
        "signal_quality_score": float(context["signal_quality_score"]),
        "duration_s": float(context["duration_s"]),
        "sample_rate_hz": float(context["sample_rate_hz"]),
        "heart_rate_bpm": float(heart_rate_metrics["heart_rate_bpm"]),
        "ibi_mean_ms": float(heart_rate_metrics["ibi_mean_ms"]),
        "sdnn_ms": float(hrv_metrics["sdnn_ms"]),
        "rmssd_ms": float(hrv_metrics["rmssd_ms"]),
        "pnn50": float(hrv_metrics["pnn50"]),
        "hrv_score": float(hrv_metrics["hrv_score"]),
        "stress_score_baseline": float(stress_metrics["stress_score"]),
        "circulation_score_baseline": float(circulation_metrics["circulation_score"]),
        "vascular_health_score_baseline": float(vascular_health_metrics["vascular_health_score"]),
        "vascular_age_estimate_baseline": float(vascular_age_metrics["vascular_age_estimate"]),
        "vascular_age_gap_baseline": float(vascular_age_metrics["vascular_age_gap"]),
        "estimated_sbp_baseline": float(blood_pressure_metrics["estimated_sbp"]),
        "estimated_dbp_baseline": float(blood_pressure_metrics["estimated_dbp"]),
        "reflection_index": float(vascular_health_metrics.get("reflection_index") or 0.0),
        "pulse_shape_score": float(vascular_health_metrics.get("pulse_shape_score") or 0.0),
        "perfusion_index": float(circulation_metrics.get("perfusion_index") or 0.0),
        "median_rise_time_s": float(circulation_metrics.get("median_rise_time_s") or 0.0),
        "median_pulse_area": float(circulation_metrics.get("median_pulse_area") or 0.0),
        "camera_available": 1.0 if camera_report.get("available") else 0.0,
        "camera_quality_score": float(camera_report.get("camera_quality_score") or 0.0),
        "camera_hr_bpm": float(camera_report.get("camera_hr_bpm") or 0.0),
        "camera_hr_difference_bpm": float(camera_report.get("heart_rate_difference_bpm") or 0.0),
        "camera_perfusion_proxy_score": float(camera_report.get("camera_perfusion_proxy_score") or 0.0),
        "camera_vascular_proxy_score": float(camera_report.get("camera_vascular_proxy_score") or 0.0),
        "camera_roi_stability_score": float(camera_report.get("roi_stability_score") or 0.0),
        "camera_rhythm_stability_score": float(camera_report.get("rhythm_stability_score") or 0.0),
        "camera_face_detection_ratio": float(camera_report.get("face_detection_ratio") or 0.0),
        "camera_frame_count": float(camera_report.get("frame_count") or 0.0),
        "camera_fusion_applied": 1.0 if camera_report.get("fusion_applied") else 0.0,
        "camera_circulation_fusion_applied": 1.0 if camera_report.get("circulation_fusion_applied") else 0.0,
        "camera_vascular_fusion_applied": 1.0 if camera_report.get("vascular_fusion_applied") else 0.0,
        "calibrated": 1.0 if blood_pressure_metrics.get("calibrated") else 0.0,
        "age": age_value,
        "sex_male": 1.0 if sex_text == "male" else 0.0,
        "sex_female": 1.0 if sex_text == "female" else 0.0,
    }
    return payload


def _predict_output(bundle: dict[str, Any], payload: dict[str, float], key: str) -> float:
    feature_order = bundle["feature_order"]
    vector = np.asarray([[float(payload.get(name, 0.0)) for name in feature_order]], dtype=float)
    model = bundle["models"][key]
    prediction = float(model.predict(vector)[0])
    return prediction


def run_ml_inference(feature_payload: dict[str, float], bundle_path: Path | None = None) -> InferenceResult:
    bundle = load_or_create_bundle(bundle_path)
    outputs = {
        "stress_score": float(np.clip(_predict_output(bundle, feature_payload, "stress_score"), 0.0, 100.0)),
        "circulation_score": float(np.clip(_predict_output(bundle, feature_payload, "circulation_score"), 0.0, 100.0)),
        "vascular_health_score": float(np.clip(_predict_output(bundle, feature_payload, "vascular_health_score"), 0.0, 100.0)),
        "vascular_age_estimate": float(np.clip(_predict_output(bundle, feature_payload, "vascular_age_estimate"), 18.0, 90.0)),
        "estimated_sbp": float(np.clip(_predict_output(bundle, feature_payload, "estimated_sbp"), 85.0, 180.0)),
        "estimated_dbp": float(np.clip(_predict_output(bundle, feature_payload, "estimated_dbp"), 50.0, 120.0)),
    }
    return InferenceResult(
        available=True,
        bundle_path=str(bundle_path or BUNDLE_PATH),
        bundle_version=str(bundle["bundle_version"]),
        bootstrap_bundle=bool(bundle.get("bootstrap_bundle")),
        outputs=outputs,
    )
