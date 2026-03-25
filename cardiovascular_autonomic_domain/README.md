# Cardiovascular and Autonomic Domain

This directory implements section 1.1 to 1.7 from `ai_model_structure_abba_s.md`.

Implemented outputs:
- 1.1 Heart rate (HR / BPM)
- 1.2 HRV metrics (SDNN, RMSSD, pNN50, HRV score, autonomic balance)
- 1.3 Stress score and state
- 1.4 Circulation score
- 1.5 Vascular health score
- 1.6 Vascular age estimate
- 1.7 Blood pressure estimate and trend

Files:
- `arduino/psl_iPPG2C_cardiovascular_autonomic/psl_iPPG2C_cardiovascular_autonomic.ino`
- `capture_and_analyze.py`
- `sequential_measurement_session.py`
- `cardiovascular_metrics.py`
- `generate_synthetic_ppg.py`

Recommended entrypoint:

```powershell
python .\run_cardiovascular_measurement.py
```

If `.venv` exists, the runner automatically re-launches itself with `C:\AI_HealthCare\.venv\Scripts\python.exe`.

Python commands below assume the project virtual environment at `C:\AI_HealthCare\.venv`.

Quick start:
1. Upload the Arduino sketch to the UNO R4 connected to the PSL-iPPG2C module.
2. Install the Python dependencies from `requirements.txt`.
3. Run a live capture:

```powershell
.\.venv\Scripts\python.exe .\cardiovascular_autonomic_domain\capture_and_analyze.py --port COM3 --duration 60 --age 42 --sex female --calibration-sbp 118 --calibration-dbp 76
```

Offline analysis:

```powershell
.\.venv\Scripts\python.exe .\cardiovascular_autonomic_domain\capture_and_analyze.py --csv-input .\cardiovascular_autonomic_domain\outputs\synthetic_capture.csv --age 42 --sex female
```

Sequential 1.1 to 1.7 run with one final output block:

```powershell
.\.venv\Scripts\python.exe .\cardiovascular_autonomic_domain\sequential_measurement_session.py --port COM3 --duration 60 --age 42 --sex female --calibration-sbp 118 --calibration-dbp 76
```

Unified interactive serial runner:

```powershell
python .\run_cardiovascular_measurement.py
```

Unified direct command example:

```powershell
python .\run_cardiovascular_measurement.py --port COM5 --duration 60 --age 42 --sex female --calibration-sbp 118 --calibration-dbp 76
```

Arduino CLI compile examples:

```powershell
& 'C:\AI_HealthCare\tools\arduino-cli\arduino-cli.exe' compile --fqbn arduino:renesas_uno:minima 'C:\AI_HealthCare\cardiovascular_autonomic_domain\arduino\psl_iPPG2C_cardiovascular_autonomic'
& 'C:\AI_HealthCare\tools\arduino-cli\arduino-cli.exe' compile --fqbn arduino:renesas_uno:unor4wifi 'C:\AI_HealthCare\cardiovascular_autonomic_domain\arduino\psl_iPPG2C_cardiovascular_autonomic'
```

Smoke test without hardware:

```powershell
.\.venv\Scripts\python.exe .\cardiovascular_autonomic_domain\generate_synthetic_ppg.py
.\.venv\Scripts\python.exe .\cardiovascular_autonomic_domain\capture_and_analyze.py --csv-input .\cardiovascular_autonomic_domain\outputs\synthetic_capture.csv --age 42 --sex female
```

Notes:
- The optional left-right channel delta term from section 1.4 is supported only if a second synchronized PPG channel is added to the CSV as `aux` or `ppg_aux_raw`.
- The blood pressure output is calibration-aware. Without cuff calibration it should be treated as a trend estimate, not a diagnostic reading.
- Vascular age is a research-style regression estimate anchored to the provided age and signal features.
