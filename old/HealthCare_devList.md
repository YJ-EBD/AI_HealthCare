# AI_HealthCare 개발 내역

- 기준일: 2026-03-23
- 로컬 저장소: `C:\AI_HealthCare`
- 원격 저장소: `https://github.com/YJ-EBD/AI_HealthCare.git`
- 기본 브랜치: `main`

## 현재 구조

- Git 루트는 `C:\AI_HealthCare`이다.
- 루트에는 `.git`, `.venv`, `old/`만 유지한다.
- 실제 프로젝트 소스와 리소스는 `old/` 아래에서 관리한다.
- 로컬 전용 파일과 캐시 데이터는 `.gitignore` 규칙으로 제외한다.

## 주요 구성

- `old/BodyCheck.py`: BodyCheck UI와 자세 측정 흐름
- `old/ai_healthcare_main.py`: 메인 UI와 화면 전환
- `old/services/analysis_service.py`: 분석 요청 처리
- `old/services/auth_service.py`: 사용자 인증과 API 키 확인
- `old/models/app_session.py`: 앱 세션 상태 관리
- `old/UI`, `old/interface`, `old/models/mediapipe`: 화면 리소스와 모델 파일

## 2026-03-23 작업 내역

- `.git`과 `.venv`를 제외한 기존 프로젝트 항목을 `old/`로 이동해 루트 구조를 정리했다.
- 루트 `.gitignore`를 복구해 `.venv`, `__pycache__`, `*.pyc`, 미리보기 이미지, `users.local.json`이 다시 소스 관리에 섞이지 않도록 정리했다.
- SOURCE CONTROL 상태를 점검해 이동된 파일 구조가 Git에 정상 반영되도록 정리했다.
- 이 문서를 현재 저장소 구조 기준의 간단한 개발 내역으로 갱신했다.

## Git 작업 원칙

- 기본 순서: `pull -> 변경 반영 -> commit -> push`
- 로컬 전용 정보는 `users.local.json`에 보관하고 저장소에는 `users.json`만 유지한다.
- 디렉토리 구조를 바꿀 때는 먼저 Git 상태를 확인한 뒤 안전하게 반영한다.
