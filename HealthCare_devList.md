# HealthCare Dev List

## 2026-03-25

- Added the new `cardiovascular_autonomic_domain/` workspace for section `1.1 ~ 1.7` of the cardiovascular and autonomic measurement flow.
- Implemented the PSL-iPPG2C Arduino UNO R4 sketch, serial capture pipeline, signal-processing logic, sequential analysis runner, and the unified root entrypoint `run_cardiovascular_measurement.py`.
- Organized the Python execution flow around the project `.venv` and recorded the installed runtime tools in `cardiovascular_autonomic_domain/List.txt`.
- Verified the new workflow with synthetic PPG smoke tests and prepared the direct serial-run path for the connected Arduino board.

## 2026-03-24

이번 채팅에서 진행한 주요 개발 내역입니다.

- `Android/` 안드로이드 앱을 기준으로 전체 화면 키오스크 UI 흐름을 정리했습니다.
- 앱 페이지 흐름을 `대기 화면 -> 성별 선택 -> 질의응답 -> 얼굴 스캔 -> 전신 스캔` 구조로 확장했습니다.
- 페이지 전환에 가로 슬라이드 애니메이션을 적용하고, 각 화면의 중앙 정렬과 1080x1920 세로 해상도 기준 배치를 보정했습니다.
- 얼굴 페이지는 얼굴 가이드 프레임, 자동 서버 연결, 카메라 준비 상태 표시가 동작하도록 정리했습니다.
- 전신 페이지는 `old/BodyCheck.py`의 전신 스켈레톤 표시 흐름을 참고해 전신 카메라와 스켈레톤 오버레이가 나오도록 연결했습니다.
- 전신 페이지의 카드, 안내 문구, 버튼, 상태 요소가 겹치지 않도록 레이아웃을 반복 보정했습니다.
- `BackEnd/`는 FastAPI 기준으로 유지하고, 기존 불필요한 Next 잔여물과 불필요한 백엔드 파일을 정리했습니다.
- FastAPI 서버는 얼굴/전신 스트림, 상태 확인, HTTP fallback 경로가 동작하도록 유지했습니다.
- 성별 선택 페이지를 참고 시안과 같은 톤으로 재구성하고, 남성/여성 카드 크기와 이미지 배치를 크게 조정했습니다.
- 성별 선택 페이지에는 사용자 제작 이미지를 포함할 수 있도록 `assets/img` 기반 자산 연결 구조를 정리했습니다.
- `MainActivity.java`에서 `file:///android_asset/` 기준 자산 경로를 주입하도록 수정해 WebView에서도 로컬 이미지가 안정적으로 보이게 했습니다.
- 성별 페이지 인물 PNG 자산을 교체하고, 최신 APK가 다시 생성되도록 여러 차례 재빌드했습니다.
- 안드로이드 앱은 Android 7.1.2 기준 immersive full screen 환경으로 동작하도록 유지했습니다.

## 현재 기준 정리

- 안드로이드 앱 경로: `Android/`
- 백엔드 경로: `BackEnd/`
- 성별 선택 이미지 자산 경로: `Android/app/src/main/assets/img/`
- 최신 APK 출력 경로: `Android/app/build/outputs/apk/debug/app-debug.apk`
- 기본 서버 포트: `8080`
