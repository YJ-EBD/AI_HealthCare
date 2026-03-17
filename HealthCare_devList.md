# AI_HealthCare 개발 내역

- 기준일: 2026-03-18
- 저장소: `C:\AI_HealthCare`
- 원격 저장소: `https://github.com/YJ-EBD/AI_HealthCare`
- 기본 브랜치: `main`

## 1. 프로젝트 개요

- `AI_HealthCare`는 PyQt5 기반 데스크톱 UI와 OpenCV/MediaPipe 기반 촬영 모듈을 결합한 헬스케어 분석 프로젝트다.
- 전체 흐름은 로그인 -> 촬영 -> 분석 -> 결과 확인 순서로 구성되어 있다.
- 촬영 데이터는 얼굴/혀 이미지를 저장한 뒤, 분석 서비스에서 OpenAI API를 호출해 결과를 생성하는 구조다.

## 2. Git / 저장소 관리

- 현재 작업 디렉토리 `C:\AI_HealthCare`를 Git 저장소로 초기화했다.
- 기본 브랜치를 `main`으로 관리한다.
- 원격 저장소 `origin`을 `https://github.com/YJ-EBD/AI_HealthCare.git`로 연결했다.
- `.gitignore`에 `.venv`, `__pycache__`, `*.pyc`, 미리보기 이미지 파일, `users.local.json`을 제외 대상으로 등록했다.
- 2026-03-18 기준으로 `main` 브랜치의 GitHub 동기화 준비를 완료했다.

## 3. 구현된 주요 기능

### 3-1. BodyCheck 자세 촬영 모듈

- `BodyCheck.py`에서 웹캠 입력을 받아 실시간 자세 측정을 수행한다.
- `UI/TypeA.png`를 기준 배경으로 사용해 본체 카메라, 얼굴 미리보기, 혀 미리보기 영역을 UI에 맞춰 합성한다.
- MediaPipe Pose를 사용해 어깨 기울기, 골반/허리 기울기, 상체 기울기, 중심 이동 값을 계산한다.
- 여러 프레임 평균값을 사용해 자세 지표를 안정적으로 표시한다.
- 얼굴 영역을 별도로 잘라 미리보기를 제공하고, 혀 색상 패턴이 감지되면 혀 이미지를 유지해서 보여준다.
- 키보드 입력으로 `Q` 종료, `R` 측정값/혀 미리보기 초기화, `F` 좌우 반전, `O` 화면 회전을 지원한다.

### 3-2. 메인 애플리케이션 UI

- `ai_healthcare_main.py`에서 `QStackedWidget` 기반 다중 페이지 UI를 구성했다.
- 로그인 페이지, 촬영 페이지, 분석 페이지, 결과 페이지 흐름을 하나의 앱에서 연결했다.
- 얼굴, 혀, 피부, 안티에이징, 건강위험도 결과 페이지 클래스를 분리해 결과 표현 구조를 갖췄다.
- 촬영된 이미지는 앱 실행 폴더 기준으로 저장하고 후속 분석 화면에서 재사용한다.

### 3-3. 인증 및 분석 서비스

- `services/auth_service.py`에서 `users.local.json` 또는 `users.json` 기반 사용자 인증을 처리한다.
- 로그인 시 사용자별 OpenAI API 키 유효성까지 확인하도록 구성했다.
- `services/analysis_service.py`에서 얼굴/혀 이미지를 Base64로 인코딩해 OpenAI API에 전달한다.
- 분석 유형별 JSON 스키마를 정의해 결과를 구조화된 형태로 받도록 구현했다.
- `models/app_session.py`에서 API 키, 촬영 이미지 경로, 분석 완료 여부 등 세션 상태를 관리한다.

## 4. 핵심 파일 역할

- `BodyCheck.py`: 실시간 자세 측정, 카메라 처리, 얼굴/혀 미리보기, UI 합성
- `ai_healthcare_main.py`: 메인 데스크톱 UI, 페이지 전환, 촬영/분석/결과 흐름
- `services/auth_service.py`: 사용자 인증 및 API 키 검증
- `services/analysis_service.py`: OpenAI 분석 요청 및 응답 파싱
- `models/app_session.py`: 앱 세션 상태 저장
- `UI/TypeA.png`: BodyCheck 전용 배경 UI 이미지
- `models/mediapipe/pose_landmarker_lite.task`: 자세 추정 모델 파일

## 5. 최근 작업 이력

- 2026-03-17: 프로젝트 초기 소스, UI 이미지, 모델 파일, 서비스 계층을 저장소에 첫 등록
- 2026-03-17: `HealthCare_devList.md` 개발내역 문서 최초 추가
- 2026-03-18: 개발내역 문서를 현재 코드 기준으로 재정리하고 GitHub 동기화 기준 문서로 갱신
- 2026-03-18: 로컬 전용 자격정보 파일(`users.local.json`)을 우선 사용하도록 정리해 비밀키가 저장소에 포함되지 않게 조정

## 6. 현재 관리 기준

- 앞으로 Git 작업은 현재 저장소 `C:\AI_HealthCare` 기준으로 진행한다.
- 사용자가 요청하면 `pull -> 변경 반영 -> commit -> push` 순서로 직접 처리한다.
- 커밋 전에는 `HealthCare_devList.md`를 함께 점검해 반영할 개발 내역이 있으면 업데이트한다.
- 실제 OpenAI API 키는 커밋 대상이 아닌 `users.local.json`에 보관하고, 저장소에는 샘플 `users.json`만 유지한다.
