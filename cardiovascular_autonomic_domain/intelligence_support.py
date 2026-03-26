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

from runtime_support import APP_DATA_DIR, log_event


EPSILON = 1e-9
MODELS_DIR = APP_DATA_DIR / "models"
BUNDLE_PATH = MODELS_DIR / "bootstrap_ml_bundle.joblib"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _score_centered(value: float | None, center: float, tolerance: float) -> float | None:
    if value is None:
        return None
    tolerance = max(float(tolerance), EPSILON)
    return float(clamp(100.0 - abs(float(value) - center) / tolerance * 100.0, 0.0, 100.0))


def _safe_nanmean(values: Any) -> float | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    with np.errstate(all="ignore"):
        value = float(np.nanmean(array))
    return value if np.isfinite(value) else None


def _heart_rate_from_peaks(peaks: np.ndarray, sample_rate_hz: float) -> float | None:
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 2 or sample_rate_hz <= 0.0:
        return None
    rr_s = np.diff(peaks) / sample_rate_hz
    rr_s = rr_s[np.isfinite(rr_s) & (rr_s > 0.0)]
    if rr_s.size == 0:
        return None
    return float(60.0 / np.mean(rr_s))


def _stat_from_group(group: Any, column: str, statistic: str = "mean") -> float | None:
    if group is None:
        return None
    try:
        value = group.loc[statistic, column]
    except Exception:
        return None
    return _safe_float(value)


_NEUROKIT2_IMPORT_ERROR: str | None = None
try:
    import neurokit2 as nk  # type: ignore[import-not-found]
except Exception as exc:  # noqa: BLE001
    nk = None
    _NEUROKIT2_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


_PYPPG_IMPORT_ERROR: str | None = None
try:
    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]

    import pyPPG  # type: ignore[import-not-found]
    from dotmap import DotMap  # type: ignore[import-not-found]
    from pyPPG.biomarkers import BmCollection  # type: ignore[import-not-found]
    from pyPPG.fiducials import FpCollection  # type: ignore[import-not-found]
    from pyPPG.ppg_sqi import get_ppgSQI  # type: ignore[import-not-found]
    from pyPPG.preproc import Preprocessing  # type: ignore[import-not-found]
except Exception as exc:  # noqa: BLE001
    pyPPG = None
    DotMap = None
    BmCollection = None
    FpCollection = None
    get_ppgSQI = None
    Preprocessing = None
    _PYPPG_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


_PYVHR_IMPORT_ERROR: str | None = None
try:
    import pyVHR  # type: ignore[import-not-found]  # noqa: F401
except Exception as exc:  # noqa: BLE001
    _PYVHR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def get_oss_runtime_status() -> dict[str, Any]:
    return {
        "neurokit2": {
            "available": nk is not None,
            "error": _NEUROKIT2_IMPORT_ERROR,
        },
        "pyppg": {
            "available": pyPPG is not None and DotMap is not None and Preprocessing is not None,
            "error": _PYPPG_IMPORT_ERROR,
        },
        "pyvhr": {
            "available": _PYVHR_IMPORT_ERROR is None,
            "error": _PYVHR_IMPORT_ERROR,
        },
    }


def _analyze_with_neurokit(signal: np.ndarray, sample_rate_hz: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "available": False,
        "engine_name": "NeuroKit2",
        "heart_rate_bpm": None,
        "peak_count": 0,
        "quality_score": None,
        "sdnn_ms": None,
        "rmssd_ms": None,
        "pnn50": None,
        "warnings": [],
        "error": None,
    }
    if nk is None:
        report["error"] = _NEUROKIT2_IMPORT_ERROR or "NeuroKit2 is unavailable."
        return report

    signal = np.asarray(signal, dtype=float)
    if signal.size < 10 or sample_rate_hz <= 0.0:
        report["warnings"].append("Signal was too short for NeuroKit2 processing.")
        return report

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cleaned = np.asarray(nk.ppg_clean(signal, sampling_rate=sample_rate_hz), dtype=float)
            _, info = nk.ppg_peaks(cleaned, sampling_rate=sample_rate_hz)
            peak_source = info.get("PPG_Peaks")
            peak_indices = np.asarray([] if peak_source is None else peak_source, dtype=int)
            quality_trace = nk.ppg_quality(cleaned, peaks=peak_indices, sampling_rate=sample_rate_hz)
            hrv_frame = nk.hrv_time(info, sampling_rate=sample_rate_hz, show=False)
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["available"] = True
    report["peak_count"] = int(peak_indices.size)
    report["heart_rate_bpm"] = _heart_rate_from_peaks(peak_indices, sample_rate_hz)
    quality_score = _safe_nanmean(quality_trace)
    report["quality_score"] = None if quality_score is None else float(clamp(quality_score * 100.0, 0.0, 100.0))

    if len(hrv_frame.index) > 0:
        row = hrv_frame.iloc[0]
        report["sdnn_ms"] = _safe_float(row.get("HRV_SDNN"))
        report["rmssd_ms"] = _safe_float(row.get("HRV_RMSSD"))
        report["pnn50"] = _safe_float(row.get("HRV_pNN50"))

    return report


def _analyze_with_pyppg(signal: np.ndarray, sample_rate_hz: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "available": False,
        "engine_name": "pyPPG",
        "sqi_score": None,
        "fiducial_count": 0,
        "biomarkers": {},
        "derived_scores": {},
        "warnings": [],
        "error": None,
    }
    if pyPPG is None or DotMap is None or Preprocessing is None:
        report["error"] = _PYPPG_IMPORT_ERROR or "pyPPG is unavailable."
        return report

    signal = np.asarray(signal, dtype=float)
    if signal.size < 32 or sample_rate_hz <= 0.0:
        report["warnings"].append("Signal was too short for pyPPG processing.")
        return report

    try:
        ppg_input = DotMap(
            start=0,
            end=int(signal.size),
            v=signal,
            fs=float(sample_rate_hz),
            name="contact_ppg",
        )
        (
            ppg_input.filt_sig,
            ppg_input.filt_d1,
            ppg_input.filt_d2,
            ppg_input.filt_d3,
        ) = Preprocessing(ppg_input, filtering=True)
        ppg_object = pyPPG.PPG(ppg_input)
        fiducials_df = FpCollection(ppg_object).get_fiducials(ppg_object)
        fiducials = pyPPG.Fiducials(fiducials_df)
        systolic_peaks = fiducials_df["sp"].dropna().astype(int).to_numpy()
        sqi_trace = get_ppgSQI(ppg_object.filt_sig, int(round(sample_rate_hz)), systolic_peaks)
        _, _, biomarker_stats = BmCollection(ppg_object, fiducials).get_biomarkers()
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["available"] = True
    report["fiducial_count"] = int(systolic_peaks.size)
    sqi_score = _safe_nanmean(sqi_trace)
    report["sqi_score"] = None if sqi_score is None else float(clamp(sqi_score * 100.0, 0.0, 100.0))

    biomarkers = {
        "tpi_s": _stat_from_group(biomarker_stats.get("ppg_sig"), "Tpi"),
        "tsys_tdia_ratio": _stat_from_group(biomarker_stats.get("sig_ratios"), "Tsys/Tdia"),
        "tsp_tpi_ratio": _stat_from_group(biomarker_stats.get("sig_ratios"), "Tsp/Tpi"),
        "ipa_ratio": _stat_from_group(biomarker_stats.get("sig_ratios"), "IPA"),
        "ai": _stat_from_group(biomarker_stats.get("derivs_ratios"), "AI"),
        "agi": _stat_from_group(biomarker_stats.get("derivs_ratios"), "AGI"),
        "agimod": _stat_from_group(biomarker_stats.get("derivs_ratios"), "AGImod"),
        "aucpi": _stat_from_group(biomarker_stats.get("ppg_sig"), "AUCpi"),
        "asp": _stat_from_group(biomarker_stats.get("ppg_sig"), "Asp"),
    }
    report["biomarkers"] = biomarkers

    tsys_tdia_score = _score_centered(biomarkers["tsys_tdia_ratio"], center=0.85, tolerance=0.35)
    tsp_tpi_score = _score_centered(biomarkers["tsp_tpi_ratio"], center=0.24, tolerance=0.12)
    ipa_score = _score_centered(biomarkers["ipa_ratio"], center=1.00, tolerance=1.00)
    ai_score = _score_centered(biomarkers["ai"], center=0.15, tolerance=0.25)
    agimod_score = _score_centered(biomarkers["agimod"], center=0.90, tolerance=1.00)

    sqi_score_value = _safe_float(report["sqi_score"]) or 0.0
    perfusion_components = [value for value in (sqi_score_value, ipa_score, tsp_tpi_score) if value is not None]
    vascular_components = [value for value in (sqi_score_value, tsys_tdia_score, ai_score, agimod_score) if value is not None]

    report["derived_scores"] = {
        "circulation_proxy_score": float(np.mean(perfusion_components)) if perfusion_components else None,
        "perfusion_index_proxy": float(
            clamp(
                0.65 * (sqi_score_value / 100.0) + 0.35 * ((ipa_score or 0.0) / 100.0),
                0.0,
                1.0,
            )
        )
        if perfusion_components
        else None,
        "vascular_proxy_score": float(np.mean(vascular_components)) if vascular_components else None,
        "ai_score": ai_score,
        "agimod_score": agimod_score,
        "tsys_tdia_score": tsys_tdia_score,
        "ipa_score": ipa_score,
        "tsp_tpi_score": tsp_tpi_score,
    }
    return report


def analyze_contact_ppg_with_oss(signal: np.ndarray, sample_rate_hz: float) -> dict[str, Any]:
    signal = np.asarray(signal, dtype=float)
    runtime_status = get_oss_runtime_status()
    neurokit_report = _analyze_with_neurokit(signal, sample_rate_hz)
    pyppg_report = _analyze_with_pyppg(signal, sample_rate_hz)

    engine_names = [
        report["engine_name"]
        for report in (neurokit_report, pyppg_report)
        if report.get("available")
    ]
    derived_signal_quality_candidates = [
        value
        for value in (
            neurokit_report.get("quality_score"),
            pyppg_report.get("sqi_score"),
        )
        if _safe_float(value) is not None
    ]

    pyppg_scores = pyppg_report.get("derived_scores") or {}
    return {
        "available": bool(engine_names),
        "engine_names": engine_names,
        "engine_label": " + ".join(engine_names) if engine_names else "미적용",
        "runtime_status": runtime_status,
        "warnings": [*neurokit_report.get("warnings", []), *pyppg_report.get("warnings", [])],
        "errors": [value for value in (neurokit_report.get("error"), pyppg_report.get("error")) if value],
        "neurokit2": neurokit_report,
        "pyppg": pyppg_report,
        "derived": {
            "signal_quality_score": float(np.mean(derived_signal_quality_candidates)) if derived_signal_quality_candidates else None,
            "circulation_proxy_score": _safe_float(pyppg_scores.get("circulation_proxy_score")),
            "perfusion_index_proxy": _safe_float(pyppg_scores.get("perfusion_index_proxy")),
            "vascular_proxy_score": _safe_float(pyppg_scores.get("vascular_proxy_score")),
            "suggested_heart_rate_bpm": _safe_float(neurokit_report.get("heart_rate_bpm")),
            "suggested_sdnn_ms": _safe_float(neurokit_report.get("sdnn_ms")),
            "suggested_rmssd_ms": _safe_float(neurokit_report.get("rmssd_ms")),
            "suggested_pnn50": _safe_float(neurokit_report.get("pnn50")),
        },
        "applied_outputs": {},
    }


def analyze_rppg_signal_with_oss(signal: np.ndarray, sample_rate_hz: float) -> dict[str, Any]:
    runtime_status = get_oss_runtime_status()
    neurokit_report = _analyze_with_neurokit(np.asarray(signal, dtype=float), sample_rate_hz)
    return {
        "available": bool(neurokit_report.get("available")),
        "engine_names": [neurokit_report["engine_name"]] if neurokit_report.get("available") else [],
        "engine_label": neurokit_report["engine_name"] if neurokit_report.get("available") else "미적용",
        "runtime_status": runtime_status,
        "heart_rate_bpm": _safe_float(neurokit_report.get("heart_rate_bpm")),
        "quality_score": _safe_float(neurokit_report.get("quality_score")),
        "peak_count": int(neurokit_report.get("peak_count") or 0),
        "warnings": list(neurokit_report.get("warnings") or []),
        "error": neurokit_report.get("error"),
        "neurokit2": neurokit_report,
    }


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
    hrv_score = np.clip(
        0.45 * ((rmssd_ms - 10.0) / 50.0 * 100.0)
        + 0.35 * ((sdnn_ms - 15.0) / 55.0 * 100.0)
        + 0.20 * (pnn50 / 25.0 * 100.0),
        0.0,
        100.0,
    )

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
    vascular_age_estimate_baseline = np.clip(
        rng.uniform(20.0, 70.0, sample_count) + 0.06 * (heart_rate_bpm - 65.0) + 0.05 * (45.0 - sdnn_ms),
        18.0,
        90.0,
    )
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
        "vascular_age_estimate": np.clip(
            age + 0.12 * (heart_rate_bpm - 68.0) + 0.08 * (50.0 - hrv_score) + 0.05 * (vascular_health_baseline - 60.0),
            18.0,
            90.0,
        ),
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
        return joblib.load(target_path)

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
    return {
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


def _predict_output(bundle: dict[str, Any], payload: dict[str, float], key: str) -> float:
    feature_order = bundle["feature_order"]
    vector = np.asarray([[float(payload.get(name, 0.0)) for name in feature_order]], dtype=float)
    model = bundle["models"][key]
    return float(model.predict(vector)[0])


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
