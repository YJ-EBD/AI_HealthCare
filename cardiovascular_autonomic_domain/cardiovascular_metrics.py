from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


EPSILON = 1e-9


@dataclass(slots=True)
class UserProfile:
    age: int | None = None
    sex: str = "unknown"
    calibration_sbp: float | None = None
    calibration_dbp: float | None = None


@dataclass(slots=True)
class SignalDataset:
    timestamps_s: np.ndarray
    ppg: np.ndarray
    beat: np.ndarray | None = None
    aux: np.ndarray | None = None
    sample_rate_hz: float = 0.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def safe_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def coeff_variation(values: np.ndarray) -> float:
    mean_value = abs(safe_mean(values))
    if mean_value <= EPSILON:
        return 0.0
    return safe_std(values) / mean_value


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return signal
    signal = signal - np.median(signal)
    scale = np.percentile(np.abs(signal), 95)
    if scale <= EPSILON:
        scale = np.std(signal)
    if scale <= EPSILON:
        return np.zeros_like(signal)
    return signal / scale


def estimate_sample_rate(timestamps_s: np.ndarray) -> float:
    if timestamps_s.size < 2:
        return 0.0
    diffs = np.diff(timestamps_s)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    return float(1.0 / np.median(diffs))


def moving_average(signal: np.ndarray, window_samples: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0 or window_samples <= 1:
        return signal.copy()
    window_samples = min(window_samples, signal.size)
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    return np.convolve(signal, kernel, mode="same")


def one_pole_lowpass(signal: np.ndarray, sample_rate_hz: float, cutoff_hz: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0 or cutoff_hz <= 0.0 or sample_rate_hz <= 0.0:
        return signal.copy()
    dt = 1.0 / sample_rate_hz
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    filtered = np.empty_like(signal)
    filtered[0] = signal[0]
    for index in range(1, signal.size):
        filtered[index] = filtered[index - 1] + alpha * (signal[index] - filtered[index - 1])
    return filtered


def one_pole_highpass(signal: np.ndarray, sample_rate_hz: float, cutoff_hz: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0 or cutoff_hz <= 0.0 or sample_rate_hz <= 0.0:
        return signal.copy()
    dt = 1.0 / sample_rate_hz
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = rc / (rc + dt)
    filtered = np.empty_like(signal)
    filtered[0] = signal[0]
    previous_input = signal[0]
    for index in range(1, signal.size):
        filtered[index] = alpha * (filtered[index - 1] + signal[index] - previous_input)
        previous_input = signal[index]
    return filtered


def zero_phase_filter(
    signal: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    mode: str,
    passes: int = 2,
) -> np.ndarray:
    filtered = np.asarray(signal, dtype=float)
    for _ in range(max(passes, 1)):
        if mode == "lowpass":
            filtered = one_pole_lowpass(filtered, sample_rate_hz, cutoff_hz)
            filtered = one_pole_lowpass(filtered[::-1], sample_rate_hz, cutoff_hz)[::-1]
        elif mode == "highpass":
            filtered = one_pole_highpass(filtered, sample_rate_hz, cutoff_hz)
            filtered = one_pole_highpass(filtered[::-1], sample_rate_hz, cutoff_hz)[::-1]
        else:
            raise ValueError(f"Unsupported filter mode: {mode}")
    return filtered


def bandpass_filter(signal: np.ndarray, sample_rate_hz: float, low_hz: float = 0.5, high_hz: float = 4.0) -> np.ndarray:
    filtered = zero_phase_filter(signal, sample_rate_hz, low_hz, mode="highpass", passes=2)
    filtered = zero_phase_filter(filtered, sample_rate_hz, high_hz, mode="lowpass", passes=2)
    return moving_average(filtered, max(1, int(sample_rate_hz * 0.03)))


def detect_systolic_peaks(filtered_ppg: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    filtered_ppg = np.asarray(filtered_ppg, dtype=float)
    if filtered_ppg.size < 3 or sample_rate_hz <= 0.0:
        return np.array([], dtype=int)

    derivative = np.diff(filtered_ppg, prepend=filtered_ppg[0])
    local_maxima = np.where(
        (filtered_ppg[1:-1] > filtered_ppg[:-2]) & (filtered_ppg[1:-1] >= filtered_ppg[2:])
    )[0] + 1
    if local_maxima.size == 0:
        return np.array([], dtype=int)

    min_distance = max(1, int(sample_rate_hz * 0.33))
    dynamic_threshold = max(
        np.percentile(filtered_ppg, 65),
        float(np.median(filtered_ppg) + 0.35 * np.std(filtered_ppg)),
    )

    selected: list[int] = []
    for peak_index in local_maxima:
        if filtered_ppg[peak_index] < dynamic_threshold:
            continue
        left = max(0, peak_index - min_distance // 2)
        right = min(filtered_ppg.size, peak_index + min_distance // 2 + 1)
        if derivative[peak_index] < -0.02 and filtered_ppg[peak_index] < np.max(filtered_ppg[left:right]):
            continue
        if not selected:
            selected.append(int(peak_index))
            continue
        if peak_index - selected[-1] < min_distance:
            if filtered_ppg[peak_index] > filtered_ppg[selected[-1]]:
                selected[-1] = int(peak_index)
        else:
            selected.append(int(peak_index))

    if len(selected) >= 3:
        return np.array(selected, dtype=int)

    relaxed_threshold = np.percentile(filtered_ppg, 55)
    selected.clear()
    for peak_index in local_maxima:
        if filtered_ppg[peak_index] < relaxed_threshold:
            continue
        if not selected or peak_index - selected[-1] >= min_distance:
            selected.append(int(peak_index))
        elif filtered_ppg[peak_index] > filtered_ppg[selected[-1]]:
            selected[-1] = int(peak_index)
    return np.array(selected, dtype=int)


def detect_beats_from_aux_channel(beat_signal: np.ndarray | None, sample_rate_hz: float) -> np.ndarray:
    if beat_signal is None or sample_rate_hz <= 0.0:
        return np.array([], dtype=int)
    beat_signal = np.asarray(beat_signal, dtype=float)
    if beat_signal.size < 3:
        return np.array([], dtype=int)
    low = float(np.percentile(beat_signal, 20))
    high = float(np.percentile(beat_signal, 80))
    if math.isclose(low, high, abs_tol=EPSILON):
        return np.array([], dtype=int)
    threshold = low + (high - low) * 0.55
    above = beat_signal >= threshold
    rising_edges = np.where(~above[:-1] & above[1:])[0] + 1
    min_distance = max(1, int(sample_rate_hz * 0.30))
    selected: list[int] = []
    for edge_index in rising_edges:
        if not selected or edge_index - selected[-1] >= min_distance:
            selected.append(int(edge_index))
    return np.array(selected, dtype=int)


def find_onsets(filtered_ppg: np.ndarray, peak_indices: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    if peak_indices.size == 0:
        return np.array([], dtype=int)
    search_back = max(1, int(sample_rate_hz * 0.45))
    onsets: list[int] = []
    last_onset = 0
    for peak_index in peak_indices:
        start = max(last_onset, peak_index - search_back)
        if start >= peak_index:
            onset = max(0, peak_index - 1)
        else:
            local_segment = filtered_ppg[start : peak_index + 1]
            onset = start + int(np.argmin(local_segment))
        if onsets and onset <= onsets[-1]:
            onset = onsets[-1] + 1
        onsets.append(min(onset, int(peak_index)))
        last_onset = onset
    return np.array(onsets, dtype=int)


def build_average_pulse(
    filtered_ppg: np.ndarray,
    onset_indices: np.ndarray,
    target_length: int = 200,
) -> np.ndarray | None:
    if onset_indices.size < 3:
        return None
    normalized_beats: list[np.ndarray] = []
    for beat_index in range(onset_indices.size - 1):
        start = int(onset_indices[beat_index])
        end = int(onset_indices[beat_index + 1])
        if end - start < 5:
            continue
        beat = filtered_ppg[start:end]
        beat = beat - beat[0]
        peak_value = float(np.max(beat))
        if peak_value <= EPSILON:
            continue
        beat = beat / peak_value
        old_axis = np.linspace(0.0, 1.0, beat.size)
        new_axis = np.linspace(0.0, 1.0, target_length)
        normalized_beats.append(np.interp(new_axis, old_axis, beat))
    if not normalized_beats:
        return None
    return np.mean(np.vstack(normalized_beats), axis=0)


def extract_pulse_features(
    filtered_ppg: np.ndarray,
    peak_indices: np.ndarray,
    onset_indices: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, Any]:
    amplitudes: list[float] = []
    rise_times_s: list[float] = []
    areas: list[float] = []
    cycle_durations_s: list[float] = []

    for beat_index in range(min(peak_indices.size, onset_indices.size - 1)):
        onset = int(onset_indices[beat_index])
        next_onset = int(onset_indices[beat_index + 1])
        peak = int(peak_indices[beat_index])
        if not (onset < peak < next_onset):
            continue

        beat_wave = filtered_ppg[onset:next_onset]
        baseline = float(filtered_ppg[onset])
        amplitude = float(filtered_ppg[peak] - baseline)
        rise_time_s = float((peak - onset) / sample_rate_hz)
        area = float(np.trapezoid(np.maximum(beat_wave - baseline, 0.0), dx=1.0 / sample_rate_hz))
        duration_s = float((next_onset - onset) / sample_rate_hz)

        if amplitude <= EPSILON or rise_time_s <= 0.0 or duration_s <= 0.0:
            continue

        amplitudes.append(amplitude)
        rise_times_s.append(rise_time_s)
        areas.append(area)
        cycle_durations_s.append(duration_s)

    return {
        "amplitudes": np.asarray(amplitudes, dtype=float),
        "rise_times_s": np.asarray(rise_times_s, dtype=float),
        "areas": np.asarray(areas, dtype=float),
        "cycle_durations_s": np.asarray(cycle_durations_s, dtype=float),
    }


def calculate_signal_quality(raw_ppg: np.ndarray, filtered_ppg: np.ndarray) -> float:
    if raw_ppg.size == 0 or filtered_ppg.size == 0:
        return 0.0
    raw_centered = raw_ppg - np.mean(raw_ppg)
    signal_energy = float(np.std(filtered_ppg))
    noise_energy = float(np.std(raw_centered - filtered_ppg))
    return clamp(100.0 * signal_energy / (signal_energy + noise_energy + EPSILON), 0.0, 100.0)


def calculate_hr_metrics(peak_indices: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    if peak_indices.size < 2:
        return {
            "heart_rate_bpm": 0.0,
            "ibi_mean_ms": 0.0,
            "ibi_median_ms": 0.0,
        }

    rr_intervals_ms = np.diff(peak_indices) / sample_rate_hz * 1000.0
    return {
        "heart_rate_bpm": float(60000.0 / np.mean(rr_intervals_ms)),
        "ibi_mean_ms": float(np.mean(rr_intervals_ms)),
        "ibi_median_ms": float(np.median(rr_intervals_ms)),
    }


def calculate_hrv_metrics(peak_indices: np.ndarray, sample_rate_hz: float) -> dict[str, float | str]:
    if peak_indices.size < 3:
        return {
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50": 0.0,
            "hrv_score": 0.0,
            "autonomic_balance_index": 0.0,
            "autonomic_balance_state": "insufficient_data",
        }

    rr_intervals_ms = np.diff(peak_indices) / sample_rate_hz * 1000.0
    rr_diff_ms = np.diff(rr_intervals_ms)
    sdnn_ms = safe_std(rr_intervals_ms)
    rmssd_ms = math.sqrt(float(np.mean(np.square(rr_diff_ms)))) if rr_diff_ms.size else 0.0
    pnn50 = 100.0 * float(np.mean(np.abs(rr_diff_ms) > 50.0)) if rr_diff_ms.size else 0.0
    heart_rate_bpm = float(60000.0 / np.mean(rr_intervals_ms))

    rmssd_score = clamp((rmssd_ms - 10.0) / 50.0 * 100.0, 0.0, 100.0)
    sdnn_score = clamp((sdnn_ms - 15.0) / 55.0 * 100.0, 0.0, 100.0)
    pnn50_score = clamp(pnn50 / 25.0 * 100.0, 0.0, 100.0)
    hrv_score = 0.45 * rmssd_score + 0.35 * sdnn_score + 0.20 * pnn50_score

    parasympathetic_ratio = rmssd_ms / max(heart_rate_bpm, 1.0)
    autonomic_balance_index = clamp(50.0 + (parasympathetic_ratio - 0.45) * 120.0, 0.0, 100.0)
    if autonomic_balance_index < 40.0:
        autonomic_balance_state = "sympathetic_dominant"
    elif autonomic_balance_index <= 60.0:
        autonomic_balance_state = "balanced"
    else:
        autonomic_balance_state = "parasympathetic_dominant"

    return {
        "sdnn_ms": float(sdnn_ms),
        "rmssd_ms": float(rmssd_ms),
        "pnn50": float(pnn50),
        "hrv_score": float(hrv_score),
        "autonomic_balance_index": float(autonomic_balance_index),
        "autonomic_balance_state": autonomic_balance_state,
    }


def calculate_stress_metrics(heart_rate_bpm: float, hrv_metrics: dict[str, float | str]) -> dict[str, float | str]:
    rmssd_ms = float(hrv_metrics["rmssd_ms"])
    sdnn_ms = float(hrv_metrics["sdnn_ms"])
    pnn50 = float(hrv_metrics["pnn50"])

    hr_component = clamp((heart_rate_bpm - 62.0) / 28.0 * 100.0, 0.0, 100.0)
    rmssd_component = clamp((40.0 - rmssd_ms) / 30.0 * 100.0, 0.0, 100.0)
    sdnn_component = clamp((45.0 - sdnn_ms) / 30.0 * 100.0, 0.0, 100.0)
    pnn50_component = clamp((15.0 - pnn50) / 15.0 * 100.0, 0.0, 100.0)
    sympathetic_ratio = heart_rate_bpm / max(rmssd_ms, 1.0)
    ratio_component = clamp((sympathetic_ratio - 1.8) / 1.3 * 100.0, 0.0, 100.0)

    stress_score = (
        0.30 * hr_component
        + 0.30 * rmssd_component
        + 0.20 * sdnn_component
        + 0.10 * pnn50_component
        + 0.10 * ratio_component
    )

    if stress_score < 33.0:
        stress_state = "stable"
    elif stress_score < 66.0:
        stress_state = "normal"
    else:
        stress_state = "tense"

    return {
        "stress_score": float(clamp(stress_score, 0.0, 100.0)),
        "stress_state": stress_state,
    }


def calculate_circulation_metrics(
    pulse_features: dict[str, Any],
    filtered_ppg: np.ndarray,
    aux_signal: np.ndarray | None,
) -> dict[str, float | bool | None]:
    amplitudes = pulse_features["amplitudes"]
    rise_times_s = pulse_features["rise_times_s"]
    areas = pulse_features["areas"]

    if amplitudes.size == 0 or rise_times_s.size == 0 or areas.size == 0:
        return {
            "circulation_score": 0.0,
            "median_amplitude": 0.0,
            "median_rise_time_s": 0.0,
            "median_pulse_area": 0.0,
            "left_right_channel_delta": None,
            "aux_channel_available": False,
            "perfusion_index": 0.0,
        }

    signal_span = float(np.max(filtered_ppg) - np.min(filtered_ppg))
    perfusion_index = float(np.median(amplitudes) / max(signal_span, EPSILON))

    amplitude_strength_score = clamp(perfusion_index * 160.0, 0.0, 100.0)
    amplitude_stability_score = clamp(100.0 - coeff_variation(amplitudes) / 0.55 * 100.0, 0.0, 100.0)
    median_rise_time_s = float(np.median(rise_times_s))
    rise_time_score = clamp(100.0 - abs(median_rise_time_s - 0.22) / 0.14 * 100.0, 0.0, 100.0)
    area_stability_score = clamp(100.0 - coeff_variation(areas) / 0.60 * 100.0, 0.0, 100.0)

    weighted_total = (
        0.35 * amplitude_strength_score
        + 0.20 * amplitude_stability_score
        + 0.25 * rise_time_score
        + 0.20 * area_stability_score
    )
    weight_sum = 1.0

    left_right_channel_delta: float | None = None
    aux_channel_available = aux_signal is not None and np.asarray(aux_signal).size == filtered_ppg.size
    if aux_channel_available:
        primary_norm = normalize_signal(filtered_ppg)
        aux_norm = normalize_signal(np.asarray(aux_signal, dtype=float))
        left_right_channel_delta = float(np.mean(np.abs(primary_norm - aux_norm)))
        channel_balance_score = clamp(100.0 - left_right_channel_delta / 0.45 * 100.0, 0.0, 100.0)
        weighted_total += 0.15 * channel_balance_score
        weight_sum += 0.15

    circulation_score = weighted_total / weight_sum

    return {
        "circulation_score": float(circulation_score),
        "median_amplitude": float(np.median(amplitudes)),
        "median_rise_time_s": median_rise_time_s,
        "median_pulse_area": float(np.median(areas)),
        "perfusion_index": perfusion_index,
        "left_right_channel_delta": left_right_channel_delta,
        "aux_channel_available": aux_channel_available,
    }


def calculate_vascular_health_metrics(
    average_pulse: np.ndarray | None,
    pulse_features: dict[str, Any],
) -> dict[str, float | None]:
    if average_pulse is None or average_pulse.size < 20:
        return {
            "vascular_health_score": 0.0,
            "dicrotic_notch_index": None,
            "reflection_index": None,
            "pulse_shape_score": 0.0,
        }

    rise_times_s = pulse_features["rise_times_s"]
    cycle_durations_s = pulse_features["cycle_durations_s"]

    systolic_limit = max(5, int(average_pulse.size * 0.45))
    systolic_peak_index = int(np.argmax(average_pulse[:systolic_limit]))

    notch_search_start = min(average_pulse.size - 3, systolic_peak_index + max(4, average_pulse.size // 20))
    notch_search_end = min(average_pulse.size - 2, int(average_pulse.size * 0.72))
    if notch_search_start >= notch_search_end:
        notch_search_start = systolic_peak_index + 1
        notch_search_end = average_pulse.size - 2

    notch_segment = average_pulse[notch_search_start:notch_search_end]
    local_minima = np.where((notch_segment[1:-1] < notch_segment[:-2]) & (notch_segment[1:-1] <= notch_segment[2:]))[0] + 1
    if local_minima.size:
        notch_index = notch_search_start + int(local_minima[0])
    else:
        notch_index = notch_search_start + int(np.argmin(notch_segment))
    dicrotic_notch_index = float(notch_index / (average_pulse.size - 1))

    secondary_search_end = min(average_pulse.size - 1, int(average_pulse.size * 0.90))
    secondary_segment = average_pulse[notch_index + 1 : secondary_search_end]
    local_maxima = np.where((secondary_segment[1:-1] > secondary_segment[:-2]) & (secondary_segment[1:-1] >= secondary_segment[2:]))[0] + 1
    if local_maxima.size:
        strongest_local_max = local_maxima[np.argmax(secondary_segment[local_maxima])]
        secondary_peak_index = notch_index + 1 + int(strongest_local_max)
        reflection_index = float(average_pulse[secondary_peak_index] / max(average_pulse[systolic_peak_index], EPSILON))
    else:
        reflection_index = float(average_pulse[notch_index] / max(average_pulse[systolic_peak_index], EPSILON))

    notch_depth = float(average_pulse[systolic_peak_index] - average_pulse[notch_index])
    notch_depth_score = clamp(notch_depth / 0.28 * 100.0, 0.0, 100.0)
    notch_timing_score = clamp(100.0 - abs(dicrotic_notch_index - 0.52) / 0.20 * 100.0, 0.0, 100.0)
    dicrotic_notch_score = 0.60 * notch_depth_score + 0.40 * notch_timing_score

    reflection_score = clamp(100.0 - abs(reflection_index - 0.40) / 0.30 * 100.0, 0.0, 100.0)
    median_rise_ratio = float(np.median(rise_times_s / np.maximum(cycle_durations_s, EPSILON))) if rise_times_s.size else 0.0
    pulse_shape_score = clamp(100.0 - abs(median_rise_ratio - 0.28) / 0.14 * 100.0, 0.0, 100.0)
    vascular_health_score = 0.40 * dicrotic_notch_score + 0.35 * reflection_score + 0.25 * pulse_shape_score

    return {
        "vascular_health_score": float(vascular_health_score),
        "dicrotic_notch_index": dicrotic_notch_index,
        "reflection_index": reflection_index,
        "pulse_shape_score": float(pulse_shape_score),
    }


def calculate_vascular_age_metrics(
    user_profile: UserProfile,
    heart_rate_bpm: float,
    hrv_metrics: dict[str, float | str],
    circulation_metrics: dict[str, float | bool | None],
    vascular_health_metrics: dict[str, float | None],
    pulse_features: dict[str, Any],
) -> dict[str, float | int | None]:
    chronological_age = user_profile.age if user_profile.age is not None else 45
    sex_adjustment = 1.5 if user_profile.sex.lower() == "male" else -1.0 if user_profile.sex.lower() == "female" else 0.0

    rmssd_ms = float(hrv_metrics["rmssd_ms"])
    sdnn_ms = float(hrv_metrics["sdnn_ms"])
    reflection_index = float(vascular_health_metrics["reflection_index"] or 0.40)
    perfusion_index = float(circulation_metrics.get("perfusion_index", 0.45) or 0.45)

    rise_times_s = pulse_features["rise_times_s"]
    cycle_durations_s = pulse_features["cycle_durations_s"]
    median_rise_ratio = float(np.median(rise_times_s / np.maximum(cycle_durations_s, EPSILON))) if rise_times_s.size else 0.28

    vascular_age_offset = 0.08 * (heart_rate_bpm - 65.0)
    vascular_age_offset += 0.08 * (45.0 - sdnn_ms)
    vascular_age_offset += 0.10 * (32.0 - rmssd_ms)
    vascular_age_offset += 14.0 * (reflection_index - 0.40)
    vascular_age_offset += 12.0 * (median_rise_ratio - 0.28)
    vascular_age_offset += 6.0 * (0.45 - perfusion_index)

    estimated_vascular_age = clamp(chronological_age + sex_adjustment + vascular_age_offset, 18.0, 90.0)
    return {
        "vascular_age_estimate": float(estimated_vascular_age),
        "vascular_age_gap": float(estimated_vascular_age - chronological_age),
        "chronological_age_reference": int(chronological_age),
    }


def calculate_blood_pressure_metrics(
    user_profile: UserProfile,
    heart_rate_bpm: float,
    circulation_metrics: dict[str, float | bool | None],
    vascular_health_metrics: dict[str, float | None],
    vascular_age_metrics: dict[str, float | int | None],
    pulse_features: dict[str, Any],
) -> dict[str, float | str | bool]:
    base_sbp = float(user_profile.calibration_sbp) if user_profile.calibration_sbp is not None else 120.0
    base_dbp = float(user_profile.calibration_dbp) if user_profile.calibration_dbp is not None else 80.0

    reflection_index = float(vascular_health_metrics["reflection_index"] or 0.40)
    perfusion_index = float(circulation_metrics.get("perfusion_index", 0.45) or 0.45)

    rise_times_s = pulse_features["rise_times_s"]
    cycle_durations_s = pulse_features["cycle_durations_s"]
    median_rise_ratio = float(np.median(rise_times_s / np.maximum(cycle_durations_s, EPSILON))) if rise_times_s.size else 0.28

    age_delta = float(vascular_age_metrics["vascular_age_gap"])
    stiffness_term = (0.28 - median_rise_ratio) / 0.08
    reflection_term = (reflection_index - 0.40) / 0.18
    hr_term = (heart_rate_bpm - 70.0) / 15.0
    perfusion_term = (0.45 - perfusion_index) / 0.20
    age_term = age_delta / 12.0

    delta_sbp = 3.5 * stiffness_term + 5.0 * reflection_term + 2.5 * hr_term + 2.0 * perfusion_term + 1.0 * age_term
    delta_dbp = 2.0 * stiffness_term + 3.0 * reflection_term + 1.5 * hr_term + 1.0 * perfusion_term + 0.8 * age_term

    estimated_sbp = clamp(base_sbp + delta_sbp, 85.0, 180.0)
    estimated_dbp = clamp(base_dbp + delta_dbp, 50.0, 120.0)

    if delta_sbp > 5.0 or delta_dbp > 4.0:
        trend = "rising"
    elif delta_sbp < -5.0 or delta_dbp < -4.0:
        trend = "falling"
    else:
        trend = "stable"

    return {
        "estimated_sbp": float(estimated_sbp),
        "estimated_dbp": float(estimated_dbp),
        "blood_pressure_trend": trend,
        "calibrated": user_profile.calibration_sbp is not None and user_profile.calibration_dbp is not None,
    }


def analyze_dataset(dataset: SignalDataset, user_profile: UserProfile | None = None) -> dict[str, Any]:
    if user_profile is None:
        user_profile = UserProfile()

    sample_rate_hz = dataset.sample_rate_hz or estimate_sample_rate(dataset.timestamps_s)
    if sample_rate_hz <= 0.0:
        raise ValueError("Unable to determine sample rate from the dataset.")

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
            warnings.append("PPG peak detection was weak, so the beat channel was used as a fallback.")

    if peak_indices.size < 3:
        raise ValueError("Not enough beats were detected to calculate HRV and vascular features.")

    onset_indices = find_onsets(filtered_ppg, peak_indices, sample_rate_hz)
    pulse_features = extract_pulse_features(filtered_ppg, peak_indices, onset_indices, sample_rate_hz)
    average_pulse = build_average_pulse(filtered_ppg, onset_indices)

    heart_rate_metrics = calculate_hr_metrics(peak_indices, sample_rate_hz)
    hrv_metrics = calculate_hrv_metrics(peak_indices, sample_rate_hz)
    stress_metrics = calculate_stress_metrics(float(heart_rate_metrics["heart_rate_bpm"]), hrv_metrics)
    circulation_metrics = calculate_circulation_metrics(pulse_features, filtered_ppg, dataset.aux)
    vascular_health_metrics = calculate_vascular_health_metrics(average_pulse, pulse_features)
    vascular_age_metrics = calculate_vascular_age_metrics(
        user_profile,
        float(heart_rate_metrics["heart_rate_bpm"]),
        hrv_metrics,
        circulation_metrics,
        vascular_health_metrics,
        pulse_features,
    )
    blood_pressure_metrics = calculate_blood_pressure_metrics(
        user_profile,
        float(heart_rate_metrics["heart_rate_bpm"]),
        circulation_metrics,
        vascular_health_metrics,
        vascular_age_metrics,
        pulse_features,
    )

    if not circulation_metrics["aux_channel_available"]:
        warnings.append("The optional left-right channel delta term is unavailable on the current single-PPG hardware.")
    if not blood_pressure_metrics["calibrated"]:
        warnings.append("Blood pressure output is trend-oriented because no personal cuff calibration was provided.")
    if user_profile.age is None:
        warnings.append("Vascular age used the default chronological age reference of 45 because no age was supplied.")

    duration_s = float(dataset.timestamps_s[-1] - dataset.timestamps_s[0]) if dataset.timestamps_s.size >= 2 else 0.0
    signal_quality_score = calculate_signal_quality(raw_ppg, filtered_ppg)
    if signal_quality_score < 35.0:
        warnings.append("Signal quality is low. Re-seat the sensor and reduce finger motion for better results.")

    return {
        "metadata": {
            "sample_rate_hz": float(sample_rate_hz),
            "sample_count": int(raw_ppg.size),
            "duration_s": duration_s,
            "signal_quality_score": float(signal_quality_score),
            "beat_detection_source": beat_source,
        },
        "heart_rate": {
            "heart_rate_bpm": float(heart_rate_metrics["heart_rate_bpm"]),
            "peak_count": int(peak_indices.size),
            "ibi_mean_ms": float(heart_rate_metrics["ibi_mean_ms"]),
            "ibi_median_ms": float(heart_rate_metrics["ibi_median_ms"]),
        },
        "hrv": hrv_metrics,
        "stress": stress_metrics,
        "circulation": circulation_metrics,
        "vascular_health": vascular_health_metrics,
        "vascular_age": vascular_age_metrics,
        "blood_pressure": blood_pressure_metrics,
        "warnings": warnings,
    }
