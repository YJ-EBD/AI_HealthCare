# Productization Blueprint for 1.1 to 1.7

This document turns the current research-style `1.1 -> 1.7` outputs into a product roadmap for a dual-modality stack:

- 4K RGB face camera
- PSL-iPPG2C finger PPG sensor

It is written to replace the current rule-based scoring in [cardiovascular_metrics.py](C:/AI_HealthCare/cardiovascular_autonomic_domain/cardiovascular_metrics.py) with a measurable, trainable, and defensible system.

## Reality Check

The current implementation is useful for prototyping, but it is not product-grade for every output:

- `1.1` and part of `1.2` can become reliable relatively quickly.
- `1.3` to `1.6` need target redefinition and supervised validation.
- `1.7` needs the most evidence, calibration logic, and no-read behavior.

No single deep learning model will make all `1.1` to `1.7` "perfect". The path to a defensible product is:

1. Redefine each output around a reference-measurable target.
2. Fuse camera and finger PPG instead of trusting either modality alone.
3. Add signal quality and uncertainty gating so the system can refuse to output numbers.
4. Train and validate against reference devices and controlled protocols.
5. Separate wellness claims from medical-device claims.

## Current Code Limitation

The present pipeline in [sequential_measurement_session.py](C:/AI_HealthCare/cardiovascular_autonomic_domain/sequential_measurement_session.py) and [cardiovascular_metrics.py](C:/AI_HealthCare/cardiovascular_autonomic_domain/cardiovascular_metrics.py) is primarily heuristic:

- `1.3 Stress` is a weighted formula derived from HR and HRV.
- `1.4 Circulation` is a handcrafted score from amplitude, rise time, and area.
- `1.5 Vascular health` is a handcrafted score from notch and reflection heuristics.
- `1.6 Vascular age` is a hand-tuned regression anchored to chronological age.
- `1.7 Blood pressure` is a rule-based delta around a cuff baseline.

These should remain as baselines, not as product claims.

## Product Redefinition

### 1.1 Heart Rate

- Product label: `Heart Rate`
- Output type: continuous regression
- Primary modality: fused `camera rPPG + iPPG2C`
- Reference: `ECG` or medical-grade pulse reference
- Recommended scope: ship in `V1`
- No-read when:
  - face ROI is unstable
  - finger PPG SQI is low
  - camera-finger disagreement exceeds threshold

### 1.2 HRV

- Product label: `Resting HRV`
- Output type: `RMSSD`, `SDNN`, optional readiness-style composite
- Primary modality: `iPPG2C` beat timing, with camera used only as secondary support
- Reference: `ECG`
- Recommended scope: ship only for `resting`, `seated or supine`, `5 min window`
- No-read when:
  - motion is detected
  - ectopic or noisy beats exceed threshold
  - acquisition is shorter than the validated window

### 1.3 Stress

- Product label: `Autonomic Arousal Index`
- Output type: low/medium/high or 0-100 regression
- Primary modality: fused `HRV + breathing proxy + vasomotor response`
- Reference: controlled task labels plus psychometric ground truth
- Recommended scope: do not market as mental health diagnosis
- No-read when:
  - HRV quality is poor
  - respiration proxy is unstable
  - user is outside validated posture or motion state

### 1.4 Circulation

- Product label: `Relative Perfusion Index`
- Output type: within-subject perfusion trend and optional perfusion map
- Primary modality: `finger PPG amplitude` plus `camera perfusion amplitude map`
- Reference: `laser Doppler perfusion` or clinical perfusion index
- Recommended scope: trend monitoring first, cross-subject absolute scoring second
- No-read when:
  - skin ROI coverage is poor
  - illumination is too low or saturated
  - finger contact is weak or drifting

### 1.5 Vascular Health

- Product label: `Arterial Stiffness / Reflection Proxy`
- Output type: regression or categorized vascular stiffness proxy
- Primary modality: fused waveform morphology from camera and finger PPG
- Reference: `cfPWV`, `baPWV`, `AIx@75`, or SphygmoCor-like arterial stiffness systems
- Recommended scope: proxy metric first, not a disease claim
- No-read when:
  - fiducial points are unstable
  - beat morphology confidence is low
  - subject is outside validated range

### 1.6 Vascular Age

- Product label: `AI-PPG Vascular Age Gap`
- Output type: estimated vascular age and delta from chronological age
- Primary modality: learned embeddings from finger PPG and camera waveform
- Reference: large cohort age labels plus longitudinal outcome validation
- Recommended scope: wellness/risk-style signal, not a diagnosis
- No-read when:
  - out-of-distribution subject detection is triggered
  - model uncertainty is high

### 1.7 Blood Pressure

- Product label: `Cuffless Blood Pressure`
- Output type: `SBP/DBP` regression
- Primary modality: fused morphology and timing features from `camera + iPPG2C`
- Reference: repeated `oscillometric cuff`, with stronger studies adding continuous arterial reference
- Recommended scope: separate project track from `1.1` to `1.6`
- No-read when:
  - calibration is missing or expired
  - subject has moved beyond the validated operating envelope
  - uncertainty exceeds threshold

## Hardware and Capture Guidance

### Camera

Use the 4K camera as a high-quality RGB sensor, but do not assume `4K at 30 fps` is always best.

- Prefer fixed exposure, fixed gain, and fixed white balance.
- Prefer stable lighting over raw resolution.
- For waveform timing tasks, `1080p at 60-120 fps` is usually more useful than `4K at 30 fps`.
- Save raw timestamps for every frame.
- Track face ROI confidence, skin coverage ratio, and motion magnitude.

### iPPG2C

- Keep the current finger PPG path as the primary morphology sensor.
- Preserve the beat channel as an auxiliary timing feature, not as the only beat source.
- Add sample-level timestamps and dropped-sample accounting.
- Capture PPG signal quality indicators online.

### Synchronization

Productization depends on synchronization.

- Use a shared host timestamp for every camera frame and every finger-PPG sample chunk.
- Store measured clock offsets.
- Keep synchronization error below the tolerance required by the BP model.

## Model Architecture

Use a dual-branch architecture instead of one monolithic model.

### Branch A: Camera rPPG

- Start with classical and open benchmarks for ROI and signal extraction.
- Use physics-informed priors first, then deep models.
- Recommended open-source starting points:
  - `rPPG-Toolbox`: https://github.com/ubicomplab/rPPG-Toolbox
  - `pyVHR`: https://github.com/phuselab/pyVHR
  - `remotebiosensing/rppg`: https://github.com/remotebiosensing/rppg

Recommended flow:

1. Face detect and track.
2. Segment skin ROI and reject poor frames.
3. Generate baseline waveforms with POS/CHROM/ICA.
4. Train a waveform-restoration network on top of the classical baseline.

### Branch B: Finger PPG

- Treat iPPG2C as the reference morphology branch inside the product.
- Recommended open-source starting points:
  - `pyPPG`: https://pubmed.ncbi.nlm.nih.gov/38478997/
  - `NeuroKit2`: https://github.com/neuropsychology/NeuroKit
  - `PaPaGei foundation model`: https://github.com/Nokia-Bell-Labs/papagei-foundation-model

Recommended flow:

1. Clean and normalize PPG.
2. Extract fiducials and beat-level biomarkers.
3. Encode waveform segments with a learned encoder or foundation model.
4. Estimate uncertainty for every beat and every window.

### Fusion Layer

Fuse camera and finger features with quality-aware attention.

Recommended input groups:

- camera waveform embedding
- finger waveform embedding
- beat timing features
- respiration proxy
- motion features
- lighting features
- demographics when justified
- calibration state for BP only

Recommended fusion models:

- temporal convolutional network for low-latency deployment
- transformer with cross-attention for richer offline models
- deep ensemble or heteroscedastic head for uncertainty

### Task Heads

- `1.1`: HR regression head
- `1.2`: beat-to-beat interval refinement plus HRV calculator
- `1.3`: arousal regression/classification head
- `1.4`: perfusion regression head
- `1.5`: stiffness/reflection proxy head
- `1.6`: vascular age head
- `1.7`: BP regression head with personalization state

## Open-Source Stack

Use open-source tools as baselines and infrastructure, not as unvalidated drop-ins.

| Purpose | Recommended open-source | Role |
| --- | --- | --- |
| Camera benchmark | `rPPG-Toolbox` | Benchmark supervised and unsupervised rPPG baselines |
| Camera classical methods | `pyVHR` | Fast baseline with POS, CHROM, ICA, MTTS-CAN and evaluation utilities |
| Finger PPG fiducials | `pyPPG` | Standardized PPG fiducials and biomarker extraction |
| Physiological preprocessing | `NeuroKit2` | HRV, signal cleaning, simulation, QA tools |
| General PPG embeddings | `PaPaGei` | Pretrained optical physiology backbone for downstream tasks |
| BP feature benchmark | `bp-benchmark` | Structured feature extraction and ML baselines for PPG-based BP |
| rPPG/CNIBP benchmark | `remotebiosensing/rppg` | Evaluation of remote PPG and non-invasive BP pipelines |
| ABP waveform baseline | `PPG2ABP` | Strong baseline for PPG-to-ABP waveform modeling |

Useful links:

- `rPPG-Toolbox`: https://github.com/ubicomplab/rPPG-Toolbox
- `pyVHR`: https://github.com/phuselab/pyVHR
- `pyPPG`: https://pubmed.ncbi.nlm.nih.gov/38478997/
- `NeuroKit2`: https://github.com/neuropsychology/NeuroKit
- `PaPaGei`: https://github.com/Nokia-Bell-Labs/papagei-foundation-model
- `bp-benchmark`: https://github.com/inventec-ai-center/bp-benchmark
- `remotebiosensing/rppg`: https://github.com/remotebiosensing/rppg
- `PPG2ABP`: https://github.com/nibtehaz/PPG2ABP

## Data Collection Protocol

You cannot productize `1.1` to `1.7` from unlabeled waveform capture alone. Collect synchronized reference data.

### Mandatory References by Output

- `1.1 Heart Rate`: ECG or validated pulse reference
- `1.2 HRV`: ECG
- `1.3 Stress`: protocol labels, psychometric labels, and preferably EDA or temperature in development studies
- `1.4 Circulation`: perfusion reference such as laser Doppler or clinical perfusion index
- `1.5 Vascular Health`: PWV or augmentation-index reference
- `1.6 Vascular Age`: large age-labeled cohort and longitudinal outcome data
- `1.7 Blood Pressure`: repeated cuff measurements, and if possible continuous arterial reference in a subset

### Subject Diversity

Build the dataset across:

- age
- sex
- BMI
- skin tone / Fitzpatrick groups
- lighting conditions
- posture
- motion level
- temperature
- cardiovascular risk spectrum

### Session Protocols

Each subject should include multiple conditions:

1. seated rest
2. paced breathing
3. postural change
4. mental stress task
5. low-light and bright-light camera conditions
6. mild motion
7. temperature challenge for perfusion
8. repeated cuff sessions for BP drift and recalibration studies

## Release Strategy

### V1

Ship only the outputs that can be defended quickly:

- `1.1 Heart Rate`
- `1.2 Resting HRV`
- `1.4 Relative Perfusion Trend`

### V1.5

Add model-based but lower-risk outputs after validation:

- `1.3 Autonomic Arousal Index`
- `1.5 Arterial Stiffness Proxy`
- `1.6 AI-PPG Vascular Age Gap`

### V2

Handle `1.7 Blood Pressure` as its own regulated workstream.

## No-Read Policy

No-read behavior is not a failure. It is a requirement.

Every output should be gated by:

- signal quality index
- motion score
- lighting score
- waveform disagreement between camera and finger PPG
- model uncertainty
- out-of-distribution detection
- calibration freshness for BP

If any gate fails, display `measurement unavailable` rather than a confident-looking number.

## Internal Validation Targets

These are recommended internal targets, not public claims.

- `1.1 Heart Rate`: rest MAE <= 2 bpm, motion MAE <= 5 bpm
- `1.2 Resting HRV`: RMSSD and SDNN only in validated resting windows, with ECG agreement target set before release
- `1.3 Arousal`: AUROC >= 0.80 against controlled stress labels
- `1.4 Perfusion`: strong within-subject trend correlation against the chosen perfusion reference
- `1.5 Stiffness Proxy`: clinically interpretable correlation against PWV or AIx
- `1.6 Vascular Age`: held-out cohort MAE target established only after pilot data
- `1.7 BP`: if marketed as BP, target the FDA-style performance envelope instead of a consumer-wellness threshold

## Regulatory Boundary

In the United States, the regulatory line matters:

- Low-risk wellness claims may fit the FDA general wellness policy.
- Blood pressure claims do not fit the same low-risk path.
- Cuffless BP should be planned as a regulated medical-device workflow.

Official references:

- FDA General Wellness guidance: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/general-wellness-policy-low-risk-devices
- FDA cuffless BP draft guidance: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cuffless-non-invasive-blood-pressure-measuring-devices-clinical-performance-testing-and-evaluation

## Immediate Engineering Actions

1. Keep the current heuristic pipeline as a baseline only.
2. Add synchronized camera capture and metadata logging.
3. Store frame timestamps, exposure, gain, lighting score, ROI confidence, and motion score.
4. Add online SQI and no-read logic before reporting any output.
5. Build a labeled dataset before training new heads.
6. Benchmark classical baselines first, then add deep models.
7. Separate `BP` into a dedicated calibration-aware pipeline.

## Recommended Repo Direction

Use the current code as the feature-engineering and smoke-test layer:

- [capture_and_analyze.py](C:/AI_HealthCare/cardiovascular_autonomic_domain/capture_and_analyze.py)
- [sequential_measurement_session.py](C:/AI_HealthCare/cardiovascular_autonomic_domain/sequential_measurement_session.py)
- [cardiovascular_metrics.py](C:/AI_HealthCare/cardiovascular_autonomic_domain/cardiovascular_metrics.py)

Then add a new multimodal stack for:

- synchronized capture
- dataset packaging
- model training
- uncertainty estimation
- subject-disjoint evaluation
- deployment calibration state

## Suggested Next Milestone

Build a `multimodal_dataset_v1` collection pipeline first. Without that dataset, training a product-grade stack for `1.1` to `1.7` is not realistic.
