# HealthCare Dev List

## 2026-03-23

이번 채팅에서 진행한 개발 내역을 간단히 정리했습니다.

- `Android/` 안드로이드 앱 프로젝트 생성
- Android 7.1.2 기준 자동 연결 구조 반영
- 앱 실행 시 서버 주소 수동 입력 없이 자동 연결되도록 수정
- WebView 기반 얼굴 카메라 수신 화면 구현
- 상단 타이틀 중심의 심플 UI 디자인 적용
- 얼굴 위치 가이드를 위한 윤곽선 오버레이 추가
- 노트북 카메라 스트림을 Android UI에서 표시하는 WebRTC 수신 흐름 구성
- `BackEnd/` 디렉토리 분리
- 기존 서버 구조를 FastAPI 기반 백엔드로 전환
- FastAPI에서 `/health`, `/offer`, `/` 경로 제공
- 카메라 WebRTC 응답 로직을 `BackEnd/worker/`로 정리
- 로컬 실행 및 WebRTC 응답 동작 확인

## 현재 기준

- Android 앱은 같은 Wi-Fi/핫스팟 게이트웨이 기준으로 자동 접속
- 백엔드는 `BackEnd/main.py` 실행 기준
- 현재 서버 포트는 `8080`
- 최근 확인된 카메라는 `1번 카메라`
