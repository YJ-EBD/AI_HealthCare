# ABBA-S AI 모델 구조 정의

본 문서는 현재 장비(4K 카메라, PSL-iPPG2C, PSL-DAQ)를 기반으로 각 측정 항목별 AI 모델 구조를 정리한 것입니다.

---

# 1. 심혈관·자율신경 영역

## 1.1 심박수 (HR)
- 입력
  - PPG Raw Signal (100~200Hz)
- 전처리
  - Bandpass Filter (0.5~4Hz)
  - Noise 제거
- 특징
  - Peak Detection
  - Beat Interval
- 출력
  - BPM (실시간 심박수)

---

## 1.2 HRV (심박변이도)
- 입력
  - PPG Beat Interval
- 특징
  - SDNN
  - RMSSD
  - pNN50
- 출력
  - HRV Score
  - 자율신경 균형 지표

---

## 1.3 스트레스 지수
- 입력
  - HR
  - HRV
- 특징
  - RMSSD 감소율
  - HR 상승 패턴
- 모델
  - Regression / Classification
- 출력
  - 스트레스 점수 (0~100)
  - 상태 (안정 / 보통 / 긴장)

---

## 1.4 혈류 순환 지수
- 입력
  - PPG Waveform
- 특징
  - Amplitude
  - Rise Time
  - Pulse Area
  - 좌우 채널 차이
- 출력
  - 혈류 순환 점수

---

## 1.5 혈관 건강 지수
- 입력
  - PPG Waveform
- 특징
  - Dicrotic Notch
  - Pulse Shape
  - Reflection Index
- 출력
  - 혈관 건강 점수

---

## 1.6 혈관 나이
- 입력
  - HRV
  - PPG 특징
  - 사용자 정보 (나이, 성별)
- 모델
  - Regression
- 출력
  - 혈관 나이 추정

---

## 1.7 혈압 추정
- 입력
  - PPG Signal
  - 개인 보정값
- 특징
  - Pulse Transit Proxy
  - Waveform Features
- 모델
  - Deep Learning Regression
- 출력
  - 혈압 추정값
  - 변화 경향

---

# 2. 피부·미용 영역

## 2.1 홍반 / 붉은기
- 입력
  - 4K 얼굴 이미지
- 전처리
  - Face Detection
  - ROI 분할
- 특징
  - RGB → Lab 변환
  - a* 채널
- 출력
  - 붉은기 점수
  - Heatmap

---

## 2.2 색소 / 톤
- 입력
  - 얼굴 이미지
- 특징
  - L* 값
  - 색 균일도
- 출력
  - 색소 점수
  - 톤 균일도

---

## 2.3 모공
- 입력
  - 고해상도 피부 이미지
- 특징
  - Texture Analysis
  - High Frequency Filter
- 모델
  - CNN Segmentation
- 출력
  - 모공 밀도
  - 평균 크기

---

## 2.4 주름
- 입력
  - 얼굴 이미지
- 특징
  - Edge Detection
  - Line Detection
- 출력
  - 주름 깊이 추정
  - 길이 및 밀도

---

## 2.5 유분·건조 경향
- 입력
  - 피부 이미지
- 특징
  - Specular Highlight
  - Texture Roughness
- 출력
  - 유분/건조 점수

---

## 2.6 여드름
- 입력
  - 얼굴 이미지
- 모델
  - Object Detection / Segmentation
- 출력
  - 여드름 개수
  - 분포

---

## 2.7 잡티
- 입력
  - 얼굴 이미지
- 특징
  - Dark Spot Detection
- 출력
  - 잡티 개수
  - 면적

---

# 3. 웰니스 해석 영역 (5장6부)

## 3.1 입력
- HRV
- 스트레스
- 혈류 지표
- 피부 상태
- 사용자 설문 (수면, 식습관 등)

## 3.2 모델
- Rule-based + ML Hybrid
- Knowledge Graph (동양의학 매핑)

## 3.3 출력
- 간 계통: 피로/해독 부담
- 심 계통: 스트레스 상태
- 비 계통: 소화 에너지
- 폐 계통: 호흡 컨디션
- 신 계통: 회복력

- 위/대장/소장/방광/담/삼초: 생활 기반 해석

---

# 4. 통합 모델 구조

## Input Fusion
- PPG 데이터
- 이미지 데이터
- 사용자 정보

## Processing
- Sensor Feature Extraction
- Vision AI
- Multi-modal Fusion

## Output
- 실시간 측정
- AI 건강 분석
- 웰니스 리포트

---

# 5. 핵심 설계 원칙

- 측정 / 추정 / 해석 분리
- 의료 표현 금지
- 데이터 표준화 필수
- 사용자 이해 중심 UI

---

# END

