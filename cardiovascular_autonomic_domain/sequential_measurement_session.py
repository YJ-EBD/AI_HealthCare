from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
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


def prepare_context(dataset: SignalDataset) -> dict[str, Any]:
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
        raise ValueError("Not enough beats were detected to calculate the 1.1 to 1.7 metrics.")

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
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    context = prepare_context(dataset)
    warnings = list(context["warnings"])

    def notify(index: int, label: str) -> None:
        if progress is not None:
            progress(index, label)

    notify(1, "1.1 Heart rate")
    heart_rate_metrics = calculate_hr_metrics(context["peak_indices"], context["sample_rate_hz"])

    notify(2, "1.2 HRV")
    hrv_metrics = calculate_hrv_metrics(context["peak_indices"], context["sample_rate_hz"])

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

    if not circulation_metrics["aux_channel_available"]:
        warnings.append("The optional left-right channel delta term is unavailable on the current single-PPG hardware.")
    if not blood_pressure_metrics["calibrated"]:
        warnings.append("Blood pressure output is trend-oriented because no personal cuff calibration was provided.")
    if user_profile.age is None:
        warnings.append("Vascular age used the default chronological age reference of 45 because no age was supplied.")
    if context["signal_quality_score"] < 35.0:
        warnings.append("Signal quality is low. Re-seat the sensor and reduce finger motion for better results.")

    return {
        "metadata": {
            "sample_rate_hz": float(context["sample_rate_hz"]),
            "sample_count": int(context["raw_ppg"].size),
            "duration_s": float(context["duration_s"]),
            "signal_quality_score": float(context["signal_quality_score"]),
            "beat_detection_source": context["beat_source"],
        },
        "heart_rate": {
            "heart_rate_bpm": float(heart_rate_metrics["heart_rate_bpm"]),
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

    lines = [
        "",
        "========================================",
        "Final consolidated result for 1.1 to 1.7",
        "========================================",
        f"Signal quality : {metadata['signal_quality_score']:.2f} / 100",
        f"1.1 Heart rate : {heart_rate['heart_rate_bpm']:.2f} bpm",
        f"1.2 HRV        : SDNN {hrv['sdnn_ms']:.2f} ms | RMSSD {hrv['rmssd_ms']:.2f} ms | pNN50 {hrv['pnn50']:.2f}% | score {hrv['hrv_score']:.2f}",
        f"1.3 Stress     : {stress['stress_score']:.2f} / 100 | {stress['stress_state']}",
        f"1.4 Circulation: {circulation['circulation_score']:.2f} / 100",
        f"1.5 Vascular   : {vascular_health['vascular_health_score']:.2f} / 100 | reflection {vascular_health['reflection_index']}",
        f"1.6 Vascular age: {vascular_age['vascular_age_estimate']:.1f} years | gap {vascular_age['vascular_age_gap']:+.1f}",
        f"1.7 Blood pressure: {blood_pressure['estimated_sbp']:.1f}/{blood_pressure['estimated_dbp']:.1f} mmHg | {blood_pressure['blood_pressure_trend']}",
    ]

    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report["warnings"])

    return "\n".join(lines)


def print_progress(step_index: int, label: str) -> None:
    print(f"[{step_index}/7] {label} complete")


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
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="Directory for captured CSV and report files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.port and not args.csv_input:
        raise SystemExit("Provide either --port for live capture or --csv-input for offline analysis.")

    output_dir = Path(args.output_dir).resolve()
    capture_path: Path | None = None

    if args.csv_input:
        print("Using existing CSV input for sequential measurement.")
        dataset = load_dataset_from_csv(Path(args.csv_input).resolve(), fallback_sample_rate_hz=args.sample_rate)
    else:
        print("Step 0/7: capturing raw PPG data. Keep the finger stable until capture finishes.")
        samples = capture_serial_session(args.port, args.baud, args.duration, args.sample_rate)
        capture_path = output_dir / "capture.csv"
        write_capture_csv(capture_path, samples)
        dataset = load_dataset_from_csv(capture_path, fallback_sample_rate_hz=args.sample_rate)
        print("Signal capture complete. Starting section 1.1 to 1.7 calculations...")

    report = run_stepwise_analysis(dataset, build_user_profile(args), progress=print_progress)
    report_path, summary_path = write_report_files(output_dir, report, capture_path=capture_path)

    print(format_console_summary(report))
    print("")
    print(f"Report JSON : {report_path}")
    print(f"Summary TXT : {summary_path}")
    if capture_path is not None:
        print(f"Capture CSV : {capture_path}")


if __name__ == "__main__":
    main()
