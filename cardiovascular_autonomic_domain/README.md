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
- `healthcare_app.py`
- `cli_tools.py`
- `runtime_support.py`
- `intelligence_support.py`
- `arduino/psl_iPPG2C_cardiovascular_autonomic/psl_iPPG2C_cardiovascular_autonomic.ino`
- `arduino/psl_iPPG2C_esp32_dwkit_v4/psl_iPPG2C_esp32_dwkit_v4.ino`
- `arduino/psl_iPPG2C_esp32_dwkit_v4/WIRING.md`
- `DEVELOPMENT_SUMMARY_1_1_TO_1_7.md`
- `PRODUCTIZATION_BLUEPRINT.md`
- `productization_targets.yaml`
- `gui_app.py`
- `camera_rppg_features.py`
- `multimodal_capture.py`
- `capture_and_analyze.py`
- `sequential_measurement_session.py`
- `cardiovascular_metrics.py`
- `generate_synthetic_ppg.py`

Recommended all-in-one entrypoint:

```powershell
python .\HealthCare.py
```

Desktop GUI entrypoint:

```powershell
python .\HealthCare.py
```

Offline camera rPPG extraction:

```powershell
python .\run_camera_rppg_extraction.py --video .\cardiovascular_autonomic_domain\outputs\synthetic_camera_test\camera_rgb.mp4 --frame-csv .\cardiovascular_autonomic_domain\outputs\synthetic_camera_test\camera_frames.csv
```

The GUI supports:

- live serial measurement with `iPPG2C`
- offline CSV analysis
- optional `camera + iPPG` multimodal dataset capture
- camera preview and camera selection
- saved `capture.csv`, `camera_rgb.mp4`, `camera_frames.csv`, and `session_manifest.json`
- automatic offline camera `rPPG` feature extraction to `camera_rppg_features.csv` and `camera_rppg_summary.json`
- all of the above from one UI

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
& 'C:\AI_HealthCare\tools\arduino-cli\arduino-cli.exe' compile --fqbn esp32:esp32:esp32 'C:\AI_HealthCare\cardiovascular_autonomic_domain\arduino\psl_iPPG2C_esp32_dwkit_v4'
```

ESP32-DWVKIT V4 wiring and upload:

- Wiring diagram: `arduino/psl_iPPG2C_esp32_dwkit_v4/WIRING.md`
- Example upload command for a classic ESP32 Dev Module on `COM8`:

```powershell
& 'C:\AI_HealthCare\tools\arduino-cli\arduino-cli.exe' compile --fqbn esp32:esp32:esp32 --upload -p COM8 'C:\AI_HealthCare\cardiovascular_autonomic_domain\arduino\psl_iPPG2C_esp32_dwkit_v4'
```

Smoke test without hardware:

```powershell
.\.venv\Scripts\python.exe .\cardiovascular_autonomic_domain\generate_synthetic_ppg.py
.\.venv\Scripts\python.exe .\cardiovascular_autonomic_domain\capture_and_analyze.py --csv-input .\cardiovascular_autonomic_domain\outputs\synthetic_capture.csv --age 42 --sex female
```

Notes:
- The implementation code is now consolidated by function inside this directory: `healthcare_app.py` for UI launch, `cli_tools.py` for CLI entrypoints, `runtime_support.py` for logging/profile/history storage, and `intelligence_support.py` for ML plus OSS signal-engine support.
- The optional left-right channel delta term from section 1.4 is supported only if a second synchronized PPG channel is added to the CSV as `aux` or `ppg_aux_raw`.
- The blood pressure output is calibration-aware. Without cuff calibration it should be treated as a trend estimate, not a diagnostic reading.
- Vascular age is a research-style regression estimate anchored to the provided age and signal features.
- The current runtime integrates open-source signal engines directly: `NeuroKit2` for contact/camera HR and HRV support, `pyPPG` for contact PPG SQI and morphology biomarkers, and `pyVHR` as an optional dependency that is only marked available when its extra runtime requirements are installed.
- `DEVELOPMENT_SUMMARY_1_1_TO_1_7.md` is the short product-planning summary for which sensor or AI combination to use for each section.
- `PRODUCTIZATION_BLUEPRINT.md` and `productization_targets.yaml` describe how to evolve the current heuristic pipeline into a multimodal product stack built around a 4K camera, PSL-iPPG2C, quality gating, supervised learning, and regulatory-aware claims.
- `camera_rppg_features.py` extracts classical camera rPPG candidate signals from recorded video using a face ROI or a center fallback ROI.
- `camera_rppg_features.py` also runs `NeuroKit2` on the selected camera signal to refine HR and camera signal quality.
- `multimodal_capture.py` stores a synchronized camera/video dataset alongside the live iPPG capture so the same session can be used later for model training.
