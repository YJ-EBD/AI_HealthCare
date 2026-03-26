# HealthCare Dev List

## 2026-03-26

### Face_AI 개발 내역
- AI Hub `한국인 피부상태 측정 데이터` 페이지와 로컬 `Face_AI/reference/한국인 피부상태 측정 데이터` 전수조사를 진행하고 데이터 규모를 공식 수치와 비교했습니다.
- 로컬 보유율을 정리했습니다: 이미지 `90.02%`, 라벨 `90.02%`, 피실험자 `90.02%`, 측정 데이터 `100%`, 메타데이터 `100%`.
- 전수조사 결과와 성능 분석 문서를 작성했습니다: `ANALYSIS_REPORT.md`, `IMPROVEMENT_PLAN.md`, `성능체크리스트.md`.
- 공식 수치 비교용 manifest와 검증 스크립트를 추가했습니다: `official_expected_manifest.json`, `executable/verify_official_equivalence.py`.
- `Face_AI/run.py` 올인원 실행 엔트리포인트를 만들고 자산 준비 후 실행되도록 정리했습니다.
- `PySide6` 기반 실시간 웹캠 UI를 추가했습니다: `live_ui.py`, `live_runtime.py`.
- UI 기본 해상도를 `900x1440` 세로 레이아웃으로 맞추고, 한글 오버레이가 `????`로 깨지지 않도록 `Pillow` 폰트 렌더링으로 수정했습니다.
- Validation 분류 평가 실행 스크립트와 런타임 평가 코드를 정리했습니다.
- 공개 체크포인트 성능을 점검했고, 원모델 평균 metric accuracy `0.760872` 및 다수 클래스 baseline 대비 약점을 확인했습니다.
- reference 분포 기반 hybrid calibration을 적용해 실사용 보정 경로를 추가했습니다. 200샘플 기준 평균 성능이 `0.779411 -> 0.816882`로 개선됐습니다.
- 항목별 추가 수집 목표를 산정했습니다: wrinkle `8,790`, sagging `3,343`, pore `4,912`, pigmentation `5,397`, dryness `1,041`.

### Git 정리 원칙
- `Face_AI/reference/`는 GitHub 업로드에서 영구 제외합니다.
- 대용량 데이터 및 생성물(`data`, `inspection`, `model/checkpoint`, `executable/output`, `__pycache__`)은 source control 안정성을 위해 제외합니다.
- 코드, 문서, 실행 스크립트 중심으로만 추적합니다.
