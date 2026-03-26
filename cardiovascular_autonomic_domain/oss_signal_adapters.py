from __future__ import annotations

from typing import Any
import warnings

import numpy as np


EPSILON = 1e-9


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
    report = {
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
    return report


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
