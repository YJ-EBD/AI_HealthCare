# PSL_Test

`PSL_iPPG2C + Arduino UNO + 4K Camera` 기반 통합 측정 프로토타입입니다.

포함 항목:
- 심박수 `HR`
- `HRV` (`SDNN`, `RMSSD`, `pNN50`, `LF/HF`)
- 혈류/순환 지수
- 혈관 건강 지수
- 스트레스 지수
- 혈관 나이
- 혈압 추정

주요 파일:
- `run.py`: Qt UI 실행
- `gui_app.py`: `PySide6/PyQt5` 기반 UI
- `serial_capture.py`: `PSL_iPPG2C` 시리얼 수집
- `camera_rppg.py`: 카메라 녹화 및 얼굴 ROI 기반 `rPPG`
- `analysis_pipeline.py`: PPG + camera 융합 분석
- `metrics.py`: 핵심 생체신호 처리 알고리즘
- `arduino/psl_iPPG2C_uno/psl_iPPG2C_uno.ino`: Arduino UNO 스케치

실행:

```powershell
.\.venv\Scripts\python.exe PSL_Test\run.py
```

권장 흐름:
1. `PSL_iPPG2C`를 Arduino UNO에 연결하고 스케치를 업로드합니다.
2. 앱에서 `Serial Port`, `Camera`를 선택합니다.
3. `PPG 측정 시작` 또는 `카메라+PPG 시작`을 누릅니다.
4. 측정 완료 후 카드/요약/결과 파일을 확인합니다.

주의:
- 혈압, 혈관 나이, 혈관 건강은 현재 연구/프로토타입용 추정치입니다.
- 실제 양산 단계에서는 임상 기준장비와 별도 검증이 필요합니다.
