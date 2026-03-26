# HealthCare 개발 내역

## 2026-03-26

1. ESP32-DWVKIT V4 기반 PSL-iPPG2C 측정 경로 추가
설명: `cardiovascular_autonomic_domain/arduino/psl_iPPG2C_esp32_dwkit_v4/` 스케치와 배선 문서를 추가하고, 기존 Python 분석 파이프라인이 그대로 읽을 수 있는 시리얼 포맷으로 맞췄습니다.

2. 1.1 ~ 1.7 제품화 전략 문서 정리
설명: `DEVELOPMENT_SUMMARY_1_1_TO_1_7.md`, `PRODUCTIZATION_BLUEPRINT.md`, `productization_targets.yaml`를 추가해 4K 카메라, PSL-iPPG2C, ML/DL 조합별 개발 방향과 제품화 범위를 정리했습니다.

3. PySide6 기반 올인원 데스크톱 UI 구현
설명: `cardiovascular_autonomic_domain/gui_app.py`를 추가하고, 측정/분석/카메라 미리보기/결과 확인/전략 문서 열람을 한 화면에서 처리할 수 있는 UI를 구현했습니다.

4. 멀티모달 데이터셋 수집 기능 추가
설명: `multimodal_capture.py`를 추가해 시리얼 iPPG와 카메라 영상을 동기화 저장하고, `capture.csv`, `camera_rgb.mp4`, `camera_frames.csv`, `session_manifest.json`이 함께 생성되도록 구성했습니다.

5. 카메라 rPPG 후보 신호 추출 기능 구현
설명: `camera_rppg_features.py`와 오프라인 추출 흐름을 추가해 저장된 영상에서 얼굴 또는 중앙 ROI 기반 `green/POS/CHROM` 신호와 카메라 HR 추정값을 뽑을 수 있도록 구성했습니다.

6. HealthCare.py 메인 실행 구조 정리
설명: 루트에 `HealthCare.py`를 추가해 전체 기능을 올인원 UI로 실행할 수 있게 정리했고, 기존 `run_cardiovascular_gui.py`와 `run_camera_rppg_extraction.py`는 보조 실행 경로로 유지했습니다.

7. UI 한국어화 및 모바일형 해상도 반영
설명: `cardiovascular_autonomic_domain/gui_app.py`를 기준으로 주요 버튼, 상태 문구, 탭 구성을 한국어로 정리하고 기본 창 크기를 `1080 x 1920`으로 맞췄습니다.

8. 카메라+iPPG 융합 분석과 신뢰도 게이트 추가
설명: `sequential_measurement_session.py`, `camera_rppg_features.py`를 확장해 카메라 HR, 관류 프록시, 혈관 프록시를 iPPG 분석에 보조 반영하고, 전체 신뢰도와 무응답 권장 로직을 함께 표시하도록 구성했습니다.

9. 제품형 런타임 기능 보강
설명: `diagnostics.py`, `product_store.py`, `multimodal_capture.py`, `capture_and_analyze.py`, `gui_app.py`를 확장해 프로필 저장, 측정 이력, 진단 로그, 시리얼/카메라 재시도, 카메라 노출·화이트밸런스·게인 설정, 장치 재연결 흐름을 추가했습니다.

10. 1.3 ~ 1.7 ML 추론 프레임워크 추가
설명: `ml_inference.py`를 추가하고 `sequential_measurement_session.py`에 연결해 스트레스, 순환, 혈관 건강, 혈관 나이, 혈압 추론이 공통 피처 페이로드 기반으로 동작하도록 정리했습니다. 현재 기본 번들은 부트스트랩 모델이며, 실측 데이터 재학습 번들로 교체 가능하게 구성했습니다.

11. 외부 오픈소스 신호 엔진 실제 연동
설명: `oss_signal_adapters.py`를 추가하고 `requirements.txt`, `camera_rppg_features.py`, `sequential_measurement_session.py`, `README.md`를 갱신해 `NeuroKit2`와 `pyPPG`를 실제 분석 경로에 연결했습니다. 접촉식 PPG는 HR/HRV/SQI/바이오마커 보강, 카메라 rPPG는 NeuroKit2 기반 HR/품질 보강이 가능하도록 정리했습니다.

12. cardiovascular_autonomic_domain 내부 기능별 코드 통합 정리
설명: `healthcare_app.py`, `cli_tools.py`, `runtime_support.py`, `intelligence_support.py`를 추가해 실행 엔트리포인트, CLI 도구, 런타임 지원, ML·오픈소스 엔진 지원을 `cardiovascular_autonomic_domain/` 내부 기능 단위로 모으고, 루트 실행 파일은 얇은 호환 래퍼로 정리했습니다.

13. 레거시 Android / BackEnd 디렉토리 제거
설명: 더 이상 사용하지 않는 `Android/`, `BackEnd/` 디렉토리를 워크스페이스와 소스컨트롤에서 제거해 현재 프로젝트 구조를 `cardiovascular_autonomic_domain` 중심으로 단순화했습니다.

## 2026-03-25

1. 심혈관·자율신경 측정 워크스페이스 구현
설명: `cardiovascular_autonomic_domain/` 디렉토리를 새로 만들고, 문서의 `1.1 ~ 1.7` 항목을 구현하기 위한 전용 작업 공간을 구성했습니다.

2. PSL-iPPG2C 신호 수집 및 분석 파이프라인 구현
설명: Arduino UNO R4용 PSL-iPPG2C 스케치, 시리얼 수집 코드, PPG 신호 필터링, 피크 검출, HR/HRV/스트레스/혈류/혈관 건강/혈관 나이/혈압 추정 로직을 구현했습니다.

3. 순차 측정 통합 실행 기능 구현
설명: `1.1 -> 1.2 -> 1.3 -> 1.4 -> 1.5 -> 1.6 -> 1.7` 순서로 계산을 진행한 뒤, 마지막에 결과를 한 번에 출력하는 통합 실행 흐름을 만들었습니다.

4. 루트 실행 진입점 구현
설명: 루트 경로에서 바로 실행할 수 있도록 `run_cardiovascular_measurement.py`를 추가했고, 시리얼 포트 선택, 측정 진행, 최종 결과 출력까지 한 파일에서 처리하도록 구성했습니다.

5. Python 가상환경 기준 실행 구조 정리
설명: 프로젝트 루트에 `.venv`를 만들고, 실행 시 `.venv`의 Python을 기준으로 동작하도록 정리했습니다. 설치한 패키지와 도구 목록은 `cardiovascular_autonomic_domain/List.txt`에 기록했습니다.

6. 측정 검증 및 문서화 작업
설명: 합성 PPG 데이터로 스모크 테스트를 진행해 결과 흐름을 검증했고, 사용 방법과 실행 예시는 `cardiovascular_autonomic_domain/README.md`에 정리했습니다.

## 2026-03-24

1. 안드로이드 키오스크 UI 흐름 정리
설명: `Android/` 앱을 기준으로 전체 화면 키오스크 흐름을 정리하고, 사용자 진행 순서를 명확하게 보이도록 구조를 재구성했습니다.

2. 앱 페이지 흐름 확장 구현
설명: 앱 흐름을 `대기 화면 -> 성별 선택 -> 질의응답 -> 얼굴 스캔 -> 전신 스캔` 구조로 확장했습니다.

3. 화면 전환 애니메이션 및 레이아웃 보정 구현
설명: 가로 슬라이드 전환 애니메이션을 적용하고, 1080x1920 세로 해상도 기준으로 중앙 정렬과 화면 배치를 보정했습니다.

4. 얼굴 스캔 화면 기능 정리
설명: 얼굴 가이드 프레임, 자동 서버 연결, 카메라 준비 상태 표시가 동작하도록 얼굴 페이지를 정리했습니다.

5. 전신 스캔 화면 연동 구현
설명: `old/BodyCheck.py`의 전신 스켈레톤 표시 흐름을 참고해 전신 카메라와 스켈레톤 오버레이가 나오도록 연결했습니다.

6. 전신 페이지 레이아웃 안정화 작업
설명: 카드, 안내 문구, 버튼, 상태 요소가 겹치지 않도록 전신 페이지 레이아웃을 반복 보정했습니다.

7. 백엔드 구조 정리
설명: `BackEnd/`는 FastAPI 기준으로 유지하고, 기존 불필요한 Next 잔여물과 불필요한 백엔드 파일을 정리했습니다.

8. 얼굴/전신 스트림 및 상태 확인 기능 유지
설명: FastAPI 서버에서 얼굴 스트림, 전신 스트림, 상태 확인, HTTP fallback 경로가 계속 동작하도록 정리했습니다.

9. 성별 선택 페이지 디자인 재구성
설명: 참고 시안과 비슷한 톤으로 성별 선택 페이지를 재구성하고, 남성/여성 카드 크기와 이미지 배치를 크게 조정했습니다.

10. 이미지 자산 연결 구조 정리
설명: 성별 선택 페이지에 사용자 제작 이미지를 포함할 수 있도록 `assets/img` 기반 자산 연결 구조를 정리했습니다.

11. WebView 로컬 자산 표시 수정
설명: `MainActivity.java`에서 `file:///android_asset/` 기준 자산 경로를 주입하도록 수정해 WebView에서도 로컬 이미지가 안정적으로 보이도록 했습니다.

12. 성별 페이지 이미지 자산 교체 및 재빌드
설명: 성별 페이지 인물 PNG 자산을 교체하고, 최신 APK가 다시 생성되도록 여러 차례 재빌드했습니다.

13. Android 7.1.2 전체화면 동작 유지
설명: 안드로이드 앱이 Android 7.1.2 기준 immersive full screen 환경에서 동작하도록 유지했습니다.

## 현재 기준 정리

1. 안드로이드 앱 경로 정리
설명: 안드로이드 앱 작업 경로는 `Android/`입니다.

2. 백엔드 경로 정리
설명: 백엔드 작업 경로는 `BackEnd/`입니다.

3. 성별 선택 이미지 자산 경로 정리
설명: 이미지 자산 경로는 `Android/app/src/main/assets/img/`입니다.

4. 최신 APK 출력 경로 정리
설명: APK 출력 경로는 `Android/app/build/outputs/apk/debug/app-debug.apk`입니다.

5. 기본 서버 포트 정리
설명: 기본 서버 포트는 `8080`입니다.
