# Face AI Runtime

`Face_AI` 는 AI Hub `한국인 피부상태 측정 데이터` 공개 산출물을 정리해,

- 공개 분류 체크포인트 기반 검증 데이터셋 평가
- `PySide6` 기반 실시간 웹캠 UI

를 바로 실행할 수 있게 만든 로컬 런타임입니다.

## 현재 구조

- `run.py`
  - 기본 진입점
  - 기본값은 `PySide6` 실시간 UI 실행
- `live_ui.py`
  - 900x1440 세로 레이아웃 웹캠 UI
  - reference 데이터 분포 기반 하이브리드 보정 모드 포함
- `live_runtime.py`
  - 얼굴 검출, 부위 crop, 체크포인트 추론 공용 런타임
- `executable/prepare_assets.py`
  - `reference` 내부 공개 산출물을 `data/`, `model/` 구조로 정리
- `executable/run_validation_classification.py`
  - 검증 데이터셋 분류 평가
- `model/checkpoint/class/released/1_2_3/`
  - 공개 분류 체크포인트
- `data/validation/images`, `data/validation/labels`
  - 검증용 이미지와 라벨
- `executable/output/`
  - 평가 결과 JSON, 실시간 스냅샷 저장 위치

## 빠른 실행

실시간 웹캠 UI 실행:

```powershell
.\.venv\Scripts\python.exe Face_AI\run.py
```

이미 자산 준비가 끝났다면:

```powershell
.\.venv\Scripts\python.exe Face_AI\run.py --skip-prepare
```

카메라 인덱스를 지정하려면:

```powershell
.\.venv\Scripts\python.exe Face_AI\run.py --camera-index 1
```

UI 기본 크기는 `900x1440` 입니다.
기본 라이브 모드에는 reference 분포 기반 보정이 켜져 있습니다.

## 검증 데이터셋 평가

기존 분류 평가 모드:

```powershell
.\.venv\Scripts\python.exe Face_AI\run.py --mode dataset-eval
```

짧게 테스트:

```powershell
.\.venv\Scripts\python.exe Face_AI\run.py --mode dataset-eval --limit 5
```

직접 평가 스크립트를 실행하려면:

```powershell
.\.venv\Scripts\python.exe Face_AI\executable\run_validation_classification.py
```

평가 결과는 `Face_AI/executable/output/classification_summary.json` 에 저장됩니다.

## 실시간 UI 설명

- `PySide6` 로 만든 세로형 대시보드입니다.
- 웹캠 화면 위에 얼굴 박스와 8개 피부 부위 박스를 그립니다.
- 아래 카드에서 다음 항목을 실시간 표시합니다.
  - 이마 주름, 이마 색소
  - 미간 주름
  - 좌우 눈가 주름
  - 좌우 볼 색소, 좌우 볼 모공
  - 입술 건조
  - 턱선 처짐
- 상단의 `요약 지수` 는 모델 클래스 결과를 `0-100` 상대 지수로 환산한 값입니다.

주의:

- 이 UI는 공개 검증용 bbox JSON 없이 웹캠 입력을 바로 처리하기 위해,
  얼굴 검출 후 데이터셋 통계 기반의 부위 crop 규칙을 사용합니다.
- 따라서 공식 검증 스크립트의 정답 비교 결과와 입력 조건이 완전히 같지는 않습니다.

## 자산 준비

필요 시 자산만 다시 정리:

```powershell
.\.venv\Scripts\python.exe Face_AI\executable\prepare_assets.py
```

학습용 이미지와 라벨까지 포함:

```powershell
.\.venv\Scripts\python.exe Face_AI\executable\prepare_assets.py --include-training
```

## 검증된 분류 평가 결과

Validation 전체 `1,391장` CPU 기준 실행 결과:

- `mean_metric_accuracy`: `0.760872`
- `dryness`: `0.971963`
- `pigmentation`: `0.836813`
- `pore`: `0.886527`
- `sagging`: `0.654206`
- `wrinkle`: `0.454851`

## 현재 한계

- 현재 즉시 실행 가능한 것은 공개 체크포인트 기반 `class` 분류입니다.
- 회귀(regression) 런타임은 공개 산출물 구조상 추가 정리가 더 필요합니다.
- 실시간 UI는 공식 bbox JSON 없이 웹캠 입력을 추론용으로 맞춘 적응 레이어입니다.

전수조사 내용은 `ANALYSIS_REPORT.md` 에 정리되어 있습니다.
