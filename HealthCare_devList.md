# HealthCare Dev List

## 2026-03-26

1. AI Hub `한국인 피부상태 측정 데이터` 페이지와 로컬 `Face_AI/reference/한국인 피부상태 측정 데이터` 전체 자료를 대조 분석했습니다.
2. 원본 대비 데이터 충족도를 확인했습니다: 이미지 `90.02%`, 라벨 `90.02%`, 피실험자 `90.02%`, 측정 데이터 `100%`, 메타데이터 `100%`.
3. `zip`, `egg`, `hwp`, `pdf`, `docx`, `csv`, `json` 자료를 직접 열어 구조와 산출물을 전수 조사했습니다.
4. 공식 수치 비교용 기준 파일과 검증 스크립트를 추가했습니다: `Face_AI/official_expected_manifest.json`, `Face_AI/executable/verify_official_equivalence.py`.
5. `Face_AI/run.py` 올인원 실행 파일을 만들어 자산 준비와 실행을 한 번에 처리할 수 있게 정리했습니다.
6. OpenCV 기반 얼굴 감지와 부위 추론 파이프라인을 정리해 실시간 분석 흐름을 연결했습니다.
7. `PySide6` 기반 세로형 실시간 UI를 개발했습니다.
8. UI 기본 해상도를 `900x1440`으로 맞추고, 실시간 웹캠 화면에서 얼굴 박스와 부위별 결과가 보이도록 구성했습니다.
9. 한글 오버레이가 `????`로 깨지는 문제를 `Pillow` 폰트 렌더링 방식으로 수정했습니다.
10. Validation 분류 평가 스크립트와 런타임 평가 코드를 정리했습니다.
11. 공개 체크포인트 성능을 점검했고, 원모델 평균 metric accuracy `0.760872`를 확인했습니다.
12. 다수 클래스 baseline과 비교해 항목별 약점을 확인했고, 특히 `wrinkle` 계열이 취약함을 정리했습니다.
13. reference 분포 기반 hybrid calibration을 추가해 실사용 보정 경로를 만들었습니다.
14. hybrid 보정 적용 시 200샘플 기준 평균 성능이 `0.779411 -> 0.816882`로 개선되는 것을 확인했습니다.
15. 추가 수집 우선순위를 산정했습니다: `wrinkle 8,790`, `sagging 3,343`, `pore 4,912`, `pigmentation 5,397`, `dryness 1,041`.
16. 분석 및 운영 문서를 작성했습니다: `Face_AI/ANALYSIS_REPORT.md`, `Face_AI/IMPROVEMENT_PLAN.md`, `Face_AI/성능체크리스트.md`, `Face_AI/README.md`.
17. Git 정리 원칙을 반영했습니다: `Face_AI/reference/`는 GitHub 업로드에서 영구 제외합니다.
18. source control 안정성을 위해 `Face_AI/data/`, `Face_AI/inspection/`, `Face_AI/model/checkpoint/`, `Face_AI/executable/output/`, `__pycache__/`도 제외 처리했습니다.
19. 코드, 문서, 실행 스크립트만 추적하도록 정리해 다른 디렉토리에 지장 없이 커밋 가능한 구조로 맞췄습니다.

------AI Hub을 참고하여 머신러닝 기반 헬스케어 디바이스 제작 예정 --------

## 2026-03-31

1. `.\.venv\Scripts\python.exe Face_AI\run.py --mode dataset-eval --skip-prepare`로 validation `1,391장` 전체를 다시 평가해 현재 CPU 기준 평균 metric accuracy `0.760704`를 확인했습니다.
2. 최신 재실행 기준 세부 metric을 다시 점검했습니다: `dryness 0.971963`, `pigmentation 0.836333`, `pore 0.886167`, `sagging 0.654206`, `wrinkle 0.454851`.
3. 생성 산출물은 계속 `Face_AI/executable/output/` 아래에서만 갱신되도록 유지했고, 해당 경로가 `Face_AI/.gitignore`로 제외되는 것을 재확인했습니다.
4. source control 안전성도 함께 다시 확인했습니다: `reference/`, `data/`, `inspection/`, `model/checkpoint/`, `executable/output/`, `__pycache__/`는 계속 추적 제외 상태로 유지되어 다른 디렉토리에 영향 없이 정리 가능한 상태입니다.
