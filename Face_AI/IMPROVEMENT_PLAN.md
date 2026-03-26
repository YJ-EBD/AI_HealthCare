# Face_AI Improvement Plan

## Basis

This plan is based on:

- AI Hub official page for dataset `71645`
- full local reference label distribution counted from `TL.zip` + `VL.zip`
- current evaluation findings in `ANALYSIS_REPORT.md`

Current local reference coverage versus the official page:

- face images: `12,545 / 13,936 = 90.02%`
- labels: `112,905 / 125,424 = 90.02%`
- subjects: `965 / 1,072 = 90.02%`
- measurement records: `84,688 / 84,688 = 100%`
- metadata: `6,432 / 6,432 = 100%`

## Why Performance Drops

The main cause is not a single runtime bug. The bigger issue is label imbalance inside the local reference package.

Examples from the full local label distribution:

- `chin_sagging`: class `0=5720`, class `6=13`
- `forehead_pigmentation`: class `1=6422`, class `5=52`
- `lip_dryness`: class `2=7631`, class `0=273`, class `4=286`
- `l_cheek_pore`: class `2=7579`, class `5=130`
- `r_cheek_pore`: class `2=7670`, class `5=130`

This means the released checkpoint can look "trained" but still underperform badly on rare grades, especially wrinkle-like tasks.

## Practical Collection Targets

The targets below are not full balancing. They are a practical floor for the next collection round:

- wrinkle and sagging: bring every class to at least `1,500`
- pigmentation and pore: bring every class to at least `1,200`
- dryness: bring every class to at least `800`

Approximate subject conversion uses the official structure of about `13 images per subject`.

### Priority 1

Wrinkle total additional labels needed: `8,790`

- forehead wrinkle: `3,153` labels, about `243` subjects
  - class `0`: `+993`
  - class `4`: `+512`
  - class `5`: `+733`
  - class `6`: `+915`
- glabellus wrinkle: `2,997` labels, about `231` subjects
  - class `3`: `+174`
  - class `4`: `+1,045`
  - class `5`: `+694`
  - class `6`: `+1,084`
- left periocular wrinkle: `1,359` labels, about `105` subjects
  - class `0`: `+616`
  - class `4`: `+343`
  - class `5`: `+213`
  - class `6`: `+187`
- right periocular wrinkle: `1,281` labels, about `99` subjects
  - class `0`: `+616`
  - class `4`: `+200`
  - class `5`: `+239`
  - class `6`: `+226`

### Priority 2

Sagging total additional labels needed: `3,343`

- chin sagging: `3,343` labels, about `258` subjects
  - class `4`: `+824`
  - class `5`: `+1,032`
  - class `6`: `+1,487`

### Priority 3

Pore total additional labels needed: `4,912`

- left cheek pore: `2,469` labels, about `190` subjects
  - class `0`: `+862`
  - class `4`: `+537`
  - class `5`: `+1,070`
- right cheek pore: `2,443` labels, about `188` subjects
  - class `0`: `+862`
  - class `4`: `+511`
  - class `5`: `+1,070`

### Priority 4

Pigmentation total additional labels needed: `5,397`

- forehead pigmentation: `2,417` labels, about `186` subjects
  - class `3`: `+160`
  - class `4`: `+1,109`
  - class `5`: `+1,148`
- left cheek pigmentation: `1,529` labels, about `118` subjects
  - class `0`: `+823`
  - class `5`: `+706`
- right cheek pigmentation: `1,451` labels, about `112` subjects
  - class `0`: `+888`
  - class `5`: `+563`

### Lowest Priority

Dryness total additional labels needed: `1,041`

- lip dryness: `1,041` labels, about `81` subjects
  - class `0`: `+527`
  - class `4`: `+514`

## Collection Order

If only one round of additional collection is possible, use this order:

1. wrinkle rare grades
2. chin sagging high grades
3. cheek pore rare grades
4. pigmentation extreme grades
5. dryness edge grades

## Retraining Pipeline

### Data prep

1. Expand training assets:

```powershell
.\.venv\Scripts\python.exe Face_AI\executable\prepare_assets.py --include-training
```

2. Keep the official `Training` and `Validation` split separate.
3. Split by subject only if an extra validation fold is needed. Do not random-split individual images across the same subject.

### Model training

1. Train one region model per official layout, same as the released package.
2. Add class-aware sampling or class-weighted loss per task.
3. Oversample the rare grades listed above first.
4. Keep the current `tolerant accuracy` style metric, but also track exact accuracy and per-class recall.

### Recommended training changes

1. Use weighted cross-entropy or focal loss for classification.
2. Add a `WeightedRandomSampler` per region/task.
3. Save confusion matrices for each task after every validation epoch.
4. Add class calibration only after raw validation improves.

### Validation policy

1. Report both:
   - exact accuracy
   - tolerant accuracy with `abs(pred - gt) < 2`
2. Report per-class recall for the rare classes:
   - wrinkle classes `4,5,6`
   - sagging classes `4,5,6`
   - pigmentation classes `4,5`
   - pore classes `4,5`
   - dryness classes `0,4`

## Current Runtime State

The live UI currently uses a `reference_hybrid` calibration mode by default because the released checkpoint is not consistently stronger than the local class priors on every task.

Use raw model only if needed:

```powershell
.\.venv\Scripts\python.exe Face_AI\run.py --skip-prepare --disable-reference-calibration
```

## Supporting Files

- `executable/output/full_label_distribution.json`
- `executable/output/data_collection_targets.json`
- `executable/output/model_training_assessment.json`
- `executable/output/sample_eval_hybrid.json`
