from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


ADC_MAX_COUNTS = 16383.0
ADC_REF_V = 5.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_beat_times(duration_s: float, heart_rate_bpm: float, variability: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beat_times: list[float] = []
    current_time = 0.0
    base_rr_s = 60.0 / heart_rate_bpm

    while current_time < duration_s + 1.0:
        respiratory_component = 0.045 * math.sin(2.0 * math.pi * 0.12 * current_time)
        random_component = rng.normal(0.0, variability)
        rr_s = clamp(base_rr_s * (1.0 + respiratory_component + random_component), 0.55, 1.50)
        current_time += rr_s
        beat_times.append(current_time)

    return np.asarray(beat_times, dtype=float)


def synthesize_ppg(duration_s: float, sample_rate_hz: float, heart_rate_bpm: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    timestamps_s = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    beat_times = build_beat_times(duration_s, heart_rate_bpm, variability=0.025, seed=seed)

    ppg = np.full_like(timestamps_s, 7800.0)
    ppg += 180.0 * np.sin(2.0 * math.pi * 0.28 * timestamps_s)
    ppg += 90.0 * np.sin(2.0 * math.pi * 0.09 * timestamps_s + 0.5)
    ppg += rng.normal(0.0, 35.0, size=timestamps_s.size)

    beat = np.full_like(timestamps_s, 1800.0)

    for beat_time in beat_times:
        ppg += 900.0 * np.exp(-0.5 * np.square((timestamps_s - beat_time) / 0.045))
        ppg -= 180.0 * np.exp(-0.5 * np.square((timestamps_s - (beat_time + 0.14)) / 0.020))
        ppg += 320.0 * np.exp(-0.5 * np.square((timestamps_s - (beat_time + 0.23)) / 0.050))
        beat += 8000.0 * np.exp(-0.5 * np.square((timestamps_s - beat_time) / 0.015))

    ppg = np.clip(ppg, 0.0, ADC_MAX_COUNTS)
    beat = np.clip(beat, 0.0, ADC_MAX_COUNTS)
    return timestamps_s, ppg, beat


def counts_to_volts(counts: np.ndarray) -> np.ndarray:
    return counts * ADC_REF_V / ADC_MAX_COUNTS


def write_csv(output_path: Path, timestamps_s: np.ndarray, ppg: np.ndarray, beat: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp_s", "sample", "ppg", "beat", "ppg_raw", "beat_raw", "ppg_v", "beat_v"],
        )
        writer.writeheader()
        ppg_v = counts_to_volts(ppg)
        beat_v = counts_to_volts(beat)
        for sample_index, timestamp_s in enumerate(timestamps_s):
            writer.writerow(
                {
                    "timestamp_s": f"{timestamp_s:.6f}",
                    "sample": sample_index,
                    "ppg": f"{ppg[sample_index]:.3f}",
                    "beat": f"{beat[sample_index]:.3f}",
                    "ppg_raw": f"{ppg[sample_index]:.3f}",
                    "beat_raw": f"{beat[sample_index]:.3f}",
                    "ppg_v": f"{ppg_v[sample_index]:.6f}",
                    "beat_v": f"{beat_v[sample_index]:.6f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic PSL-iPPG2C capture for smoke testing.")
    parser.add_argument("--duration", type=float, default=60.0, help="Synthetic capture length in seconds.")
    parser.add_argument("--sample-rate", type=float, default=200.0, help="Sample rate in Hz.")
    parser.add_argument("--heart-rate", type=float, default=72.0, help="Target average heart rate in bpm.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "outputs" / "synthetic_capture.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    timestamps_s, ppg, beat = synthesize_ppg(args.duration, args.sample_rate, args.heart_rate, args.seed)
    output_path = Path(args.output).resolve()
    write_csv(output_path, timestamps_s, ppg, beat)

    print(f"Synthetic capture written to {output_path}")


if __name__ == "__main__":
    main()
