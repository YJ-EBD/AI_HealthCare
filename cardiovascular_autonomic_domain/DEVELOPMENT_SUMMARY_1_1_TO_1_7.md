# 1.1 to 1.7 Development Summary

This file is the short version of the product plan for the current hardware stack:

- 4K RGB camera
- PSL-iPPG2C finger PPG sensor
- Open-source machine learning or deep learning

## Recommended Combination by Output

| Section | Product Output | Main Sensor | Optional Sensor | AI Role | Recommended Release Shape |
| --- | --- | --- | --- | --- | --- |
| `1.1` | Heart rate | `iPPG2C` | Camera | Signal quality and fusion only if needed | `V1` |
| `1.2` | Resting HRV | `iPPG2C` | Camera | Beat cleanup, no-read, artifact control | `V1` |
| `1.3` | Autonomic arousal | `iPPG2C` | Camera | Core ML/DL task | `V1.5` |
| `1.4` | Relative perfusion | `iPPG2C` | Camera | Trend regression and robustness | `V1` |
| `1.5` | Arterial stiffness proxy | `iPPG2C` | Camera | Core ML/DL task | `V1.5` |
| `1.6` | AI-PPG vascular age gap | `iPPG2C` | Camera | Core DL biomarker task | `V1.5` |
| `1.7` | Cuffless blood pressure | `iPPG2C + Camera` | None | Core ML/DL plus personalization | `V2` |

## How to Build It

### V1

Deliver the most defensible outputs first:

- `1.1 Heart rate`
- `1.2 Resting HRV`
- `1.4 Relative perfusion trend`

Implementation style:

- use `iPPG2C` as the main production sensor
- keep the current signal-processing pipeline as the baseline
- add a desktop UI, signal quality checks, session logging, and no-read behavior

### V1.5

Add model-based outputs:

- `1.3 Autonomic arousal`
- `1.5 Arterial stiffness proxy`
- `1.6 AI-PPG vascular age gap`

Implementation style:

- train on synchronized `iPPG2C + camera` data
- start from open-source models and benchmark them against the current baseline
- expose these as research or wellness outputs before making stronger claims

### V2

Handle blood pressure as a separate workstream:

- `1.7 Cuffless blood pressure`

Implementation style:

- use `iPPG2C + camera + calibration + ML/DL`
- keep strict no-read logic
- validate against cuff measurements before release

## Open-Source Starting Points

- `rPPG-Toolbox`: camera rPPG benchmarking
- `pyVHR`: classical camera rPPG baselines
- `pyPPG`: finger PPG fiducials and biomarkers
- `NeuroKit2`: physiological preprocessing and HRV utilities
- `PaPaGei`: learned optical-physiology embeddings
- `bp-benchmark` and `PPG2ABP`: blood-pressure research baselines

## Current Implementation Status

Today the repository already supports:

- serial capture from `iPPG2C`
- sequential `1.1` to `1.7` analysis
- saved JSON/TXT/CSV outputs
- rule-based baseline metrics

The next step is to add:

- a desktop UI
- camera session management
- synchronized multimodal dataset packaging
- model inference hooks for `1.3` to `1.7`
