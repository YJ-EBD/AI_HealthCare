from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
from serial.tools import list_ports

from diagnostics import LOG_PATH, configure_logging, log_event, log_exception
from capture_and_analyze import (
    build_user_profile,
    capture_serial_session,
    load_dataset_from_csv,
    write_capture_csv,
    write_report_files,
)
from camera_rppg_features import extract_camera_rppg_features
from multimodal_capture import capture_multimodal_session, open_camera_capture, probe_camera_indices
from product_store import ProductStore
from sequential_measurement_session import format_console_summary, load_camera_summary, run_stepwise_analysis

try:
    from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
    from PySide6.QtGui import QAction, QImage, QPixmap, QTextOption
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"
except ImportError:
    from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt5.QtGui import QImage, QPixmap, QTextOption
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PyQt5"


ROOT_DIR = Path(__file__).resolve().parent.parent
DOMAIN_DIR = ROOT_DIR / "cardiovascular_autonomic_domain"
OUTPUTS_DIR = DOMAIN_DIR / "outputs"
SUMMARY_MD_PATH = DOMAIN_DIR / "DEVELOPMENT_SUMMARY_1_1_TO_1_7.md"
BLUEPRINT_MD_PATH = DOMAIN_DIR / "PRODUCTIZATION_BLUEPRINT.md"

STRATEGY_SUMMARY_TEXT = """# 1.1 ~ 1.7 개발 요약

현재 하드웨어 구성:
- 4K RGB 카메라
- PSL-iPPG2C 손가락 PPG 센서
- 오픈소스 머신러닝 / 딥러닝

권장 조합:
- 1.1 심박수: iPPG2C 중심, 필요 시 카메라 보조
- 1.2 안정 시 HRV: iPPG2C 중심
- 1.3 자율신경 각성: iPPG2C + 카메라 + 머신러닝/딥러닝
- 1.4 상대 관류: iPPG2C 중심, 카메라 보조
- 1.5 혈관 강직도 프록시: iPPG2C + 카메라 + 머신러닝/딥러닝
- 1.6 AI-PPG 혈관 나이 차이: iPPG2C + 카메라 + 딥러닝
- 1.7 비커프 혈압: iPPG2C + 카메라 + 개인 보정 + 머신러닝/딥러닝

출시 권장 순서:

V1
- 1.1 심박수
- 1.2 안정 시 HRV
- 1.4 상대 관류 추세

V1.5
- 1.3 자율신경 각성
- 1.5 혈관 강직도 프록시
- 1.6 AI-PPG 혈관 나이 차이

V2
- 1.7 비커프 혈압

현재 저장소에서 지원하는 것:
- iPPG2C 시리얼 수집
- 1.1 ~ 1.7 순차 분석
- JSON / TXT / CSV 결과 저장
- 카메라 + iPPG 멀티모달 세션 저장
- 카메라 rPPG 특징 추출

다음 단계:
- 카메라 세션 관리 강화
- 동기화된 멀티모달 데이터셋 고정
- 1.3 ~ 1.7용 모델 추론 연결
- 무응답 정책과 품질 게이트 강화
"""

STRATEGY_BLUEPRINT_TEXT = """# 1.1 ~ 1.7 제품화 청사진

현 구조는 연구용 프로토타입에는 유용하지만, 모든 항목을 그대로 제품 주장으로 쓰기에는 부족합니다.

핵심 원칙:
1. 각 출력값을 기준 장비로 검증 가능한 목표로 다시 정의합니다.
2. 카메라와 손가락 PPG를 함께 쓰되, 품질이 낮으면 값을 내지 않습니다.
3. 오픈소스 모델은 출발점으로만 쓰고, 자체 데이터로 재학습과 검증을 합니다.
4. 웰니스 주장과 의료기기 주장을 분리합니다.

항목별 방향:

1.1 심박수
- 출력: 연속 회귀
- 주 센서: iPPG2C
- 보조 센서: 카메라
- 기준값: ECG 또는 의료급 맥박 기준
- 무응답 조건: 얼굴 ROI 불안정, 손가락 PPG 품질 저하, 센서 간 불일치

1.2 HRV
- 출력: RMSSD, SDNN, 필요 시 준비도 지표
- 주 센서: iPPG2C 비트 타이밍
- 기준값: ECG
- 권장 조건: 안정 시, 앉은 자세 또는 누운 자세, 5분 창
- 무응답 조건: 움직임, 잡음성 박동 과다, 검증 창보다 짧은 수집 시간

1.3 자율신경 각성
- 출력: 저/중/고 또는 0~100 회귀
- 조합: HRV + 호흡 프록시 + 혈관운동 반응
- 기준값: 과제 라벨 + 설문 라벨
- 주의: 정신건강 진단으로 홍보하지 않음

1.4 상대 관류
- 출력: 개인 내 관류 추세, 선택적으로 관류 맵
- 조합: 손가락 PPG 진폭 + 카메라 관류 진폭 맵
- 기준값: 레이저 도플러 관류 또는 임상 관류 지수
- 권장: 절대 점수보다 추세 모니터링부터 시작

1.5 혈관 건강 프록시
- 출력: 동맥 강직도 / 반사파 프록시
- 조합: 카메라 + iPPG2C 파형 형태 융합
- 기준값: cfPWV, baPWV, AIx@75 등
- 주의: 질환 진단 대신 프록시 지표로 사용

1.6 혈관 나이
- 출력: 추정 혈관 나이와 실제 나이 대비 차이
- 조합: 카메라 + iPPG2C 임베딩 기반 딥러닝
- 기준값: 대규모 연령 라벨 코호트 + 장기 추적 검증
- 권장: 웰니스 / 위험 신호 용도

1.7 혈압
- 출력: SBP / DBP 회귀
- 조합: 카메라 + iPPG2C + 개인 보정 + 머신러닝/딥러닝
- 기준값: 반복 커프 측정, 가능하면 연속 동맥 기준
- 주의: 1.1 ~ 1.6과 별도 규제 트랙으로 관리

하드웨어 가이드:
- 카메라는 4K 해상도 자체보다 고정 노출, 고정 화이트밸런스, 안정적인 조명이 더 중요합니다.
- 파형 타이밍이 중요하면 4K 30fps보다 1080p 60~120fps가 더 유리할 수 있습니다.
- iPPG2C는 주 파형 센서로 유지하고, 비트 채널은 보조 타이밍 정보로 활용합니다.
- 모든 카메라 프레임과 PPG 샘플에 공통 시간축을 저장해야 합니다.

권장 오픈소스 출발점:
- rPPG-Toolbox: 카메라 rPPG 벤치마크
- pyVHR: 고전적 카메라 rPPG 기준선
- pyPPG: 손가락 PPG 지표 추출
- NeuroKit2: 생리신호 전처리와 HRV 유틸리티
- PaPaGei: 광학 생리 임베딩 기반 모델
- bp-benchmark, PPG2ABP: 혈압 연구 기준선

모델 구조 권장:
- 카메라 분기: 얼굴 추적, 피부 ROI, POS/CHROM 기준선, 파형 복원 모델
- 손가락 PPG 분기: 필터링, 특징점 추출, 비트 품질 추정
- 융합 계층: 품질 가중 어텐션 또는 시계열 모델
- 태스크 헤드: 1.1~1.7별 개별 헤드

필수 데이터 수집:
- 1.1 심박수: ECG
- 1.2 HRV: ECG
- 1.3 자율신경 각성: 과제 라벨, 설문, 가능하면 EDA/온도
- 1.4 관류: 관류 기준 장비
- 1.5 혈관 강직도: PWV 또는 AIx 기준 장비
- 1.6 혈관 나이: 대규모 연령 라벨 데이터
- 1.7 혈압: 반복 커프 측정

무응답 정책:
- 신호 품질 지수 부족
- 움직임 과다
- 조명 불량
- 카메라와 손가락 PPG 파형 불일치
- 모델 불확실성 과다
- BP의 경우 보정 만료

즉시 권장 작업:
1. 현재 규칙 기반 파이프라인은 기준선으로 유지합니다.
2. 동기화된 카메라 + iPPG 수집과 메타데이터 저장을 강화합니다.
3. 온라인 SQI와 무응답 로직을 먼저 넣습니다.
4. 오픈소스 기준선을 먼저 비교하고, 그다음 자체 모델을 학습합니다.
5. 혈압은 별도 보정형 파이프라인으로 분리합니다.
"""


def list_serial_port_rows() -> list[object]:
    return list(list_ports.comports())


def describe_serial_port(port: object) -> str:
    return f"{port.device} - 시리얼 장치"


def describe_camera_entry(camera_info: dict[str, int | float]) -> str:
    width = int(camera_info.get("width", 0))
    height = int(camera_info.get("height", 0))
    fps = float(camera_info.get("fps", 0.0))
    fps_text = f"{fps:.1f} fps" if fps > 0.0 else "fps 알 수 없음"
    return f"카메라 {camera_info['index']} - {width}x{height} / {fps_text}"


def translate_camera_signal_name(signal_name: str) -> str:
    mapping = {
        "green": "녹색 채널",
        "pos": "POS",
        "chrom": "CHROM",
    }
    return mapping.get(signal_name, signal_name)


def try_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def try_int(value: str) -> int | None:
    number = try_float(value)
    return None if number is None else int(number)


@dataclass(slots=True)
class SessionConfig:
    port: str | None
    baud: int
    duration_s: float
    sample_rate_hz: float
    age: int | None
    sex: str
    calibration_sbp: float | None
    calibration_dbp: float | None
    csv_input: Path | None = None
    use_camera_dataset: bool = False
    camera_index: int | None = None
    camera_width: int | None = None
    camera_height: int | None = None
    camera_fps: float | None = None
    camera_auto_exposure: bool | None = True
    camera_exposure_value: float | None = None
    camera_auto_white_balance: bool | None = True
    camera_white_balance_value: float | None = None
    camera_gain_value: float | None = None
    serial_retry_count: int = 2
    camera_retry_count: int = 2
    reconnect_enabled: bool = True
    no_data_timeout_s: float = 5.0
    profile_id: int | None = None
    profile_name: str = ""
    profile_notes: str = ""


class MeasurementWorker(QObject):
    progress_text = Signal(str)
    result_ready = Signal(dict, str, str, str, dict)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, config: SessionConfig) -> None:
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        output_dir = OUTPUTS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        capture_path: Path | None = None
        extra_paths: dict[str, str] = {"output_dir": str(output_dir), "log_path": str(LOG_PATH)}
        camera_summary: dict[str, Any] | None = None

        try:
            log_event(
                "measurement_worker",
                "측정 작업을 시작합니다.",
                details={
                    "mode": "csv" if self.config.csv_input else "multimodal" if self.config.use_camera_dataset else "live",
                    "port": self.config.port,
                    "profile_id": self.config.profile_id,
                },
            )
            if self.config.csv_input is not None:
                self.progress_text.emit("기존 CSV 입력을 불러와 분석합니다.")
                dataset = load_dataset_from_csv(self.config.csv_input, fallback_sample_rate_hz=self.config.sample_rate_hz)
                auto_camera_summary_path = self.config.csv_input.parent / "camera_rppg_summary.json"
                camera_summary = load_camera_summary(auto_camera_summary_path)
                if camera_summary is not None:
                    extra_paths["camera_summary_json"] = str(auto_camera_summary_path)
                    self.progress_text.emit("같은 폴더의 카메라 요약을 불러와 카메라+iPPG 융합 분석을 적용합니다.")
            elif self.config.use_camera_dataset:
                if not self.config.port:
                    raise ValueError("라이브 멀티모달 수집을 하려면 시리얼 포트가 필요합니다.")
                if self.config.camera_index is None:
                    raise ValueError("멀티모달 수집을 하려면 카메라를 선택해야 합니다.")

                self.progress_text.emit("멀티모달 수집을 시작합니다: 카메라 + iPPG")
                capture_bundle = capture_multimodal_session(
                    port=self.config.port,
                    baud=self.config.baud,
                    duration_s=self.config.duration_s,
                    fallback_sample_rate_hz=self.config.sample_rate_hz,
                    output_dir=output_dir,
                    camera_index=self.config.camera_index,
                    camera_width=self.config.camera_width,
                    camera_height=self.config.camera_height,
                    camera_fps=self.config.camera_fps,
                    camera_auto_exposure=self.config.camera_auto_exposure,
                    camera_exposure_value=self.config.camera_exposure_value,
                    camera_auto_white_balance=self.config.camera_auto_white_balance,
                    camera_white_balance_value=self.config.camera_white_balance_value,
                    camera_gain_value=self.config.camera_gain_value,
                    camera_retry_count=self.config.camera_retry_count,
                    serial_retry_count=self.config.serial_retry_count,
                    reconnect_enabled=self.config.reconnect_enabled,
                    no_data_timeout_s=self.config.no_data_timeout_s,
                    status_callback=self.progress_text.emit,
                )
                capture_path = Path(capture_bundle["capture_csv_path"])
                extra_paths.update({
                    "camera_video": str(capture_bundle["video_path"]),
                    "camera_frames_csv": str(capture_bundle["frame_csv_path"]),
                    "session_manifest": str(capture_bundle["manifest_path"]),
                })
                self.progress_text.emit("녹화된 영상에서 카메라 rPPG 특징을 추출합니다.")
                camera_features = extract_camera_rppg_features(
                    Path(capture_bundle["video_path"]),
                    output_dir,
                    frame_timestamps_path=Path(capture_bundle["frame_csv_path"]),
                    status_callback=self.progress_text.emit,
                )
                extra_paths["camera_features_csv"] = str(camera_features["features_csv_path"])
                extra_paths["camera_summary_json"] = str(camera_features["summary_json_path"])
                camera_summary_candidate = camera_features.get("summary")
                camera_summary = camera_summary_candidate if isinstance(camera_summary_candidate, dict) else None
                dataset = load_dataset_from_csv(capture_path, fallback_sample_rate_hz=self.config.sample_rate_hz)
                self.progress_text.emit("멀티모달 수집이 끝났습니다. 1.1 ~ 1.7 분석을 시작합니다.")
            else:
                if not self.config.port:
                    raise ValueError("CSV 분석이 아니라면 시리얼 포트가 필요합니다.")
                self.progress_text.emit("0/7 단계: 원시 PPG 데이터를 수집합니다.")
                samples = capture_serial_session(
                    self.config.port,
                    self.config.baud,
                    self.config.duration_s,
                    self.config.sample_rate_hz,
                    status_callback=self.progress_text.emit,
                    retry_count=self.config.serial_retry_count,
                    no_data_timeout_s=self.config.no_data_timeout_s,
                )
                capture_path = output_dir / "capture.csv"
                write_capture_csv(capture_path, samples)
                dataset = load_dataset_from_csv(capture_path, fallback_sample_rate_hz=self.config.sample_rate_hz)
                self.progress_text.emit("신호 수집이 끝났습니다. 1.1 ~ 1.7 분석을 시작합니다.")

            profile_args = SimpleNamespace(
                age=self.config.age,
                sex=self.config.sex,
                calibration_sbp=self.config.calibration_sbp,
                calibration_dbp=self.config.calibration_dbp,
            )
            report = run_stepwise_analysis(
                dataset,
                build_user_profile(profile_args),
                camera_summary=camera_summary,
                progress=lambda index, label: self.progress_text.emit(f"[{index}/7] {label} 완료"),
            )
            report_path, summary_path = write_report_files(output_dir, report, capture_path=capture_path)
            summary_text = format_console_summary(report)
            log_event(
                "measurement_worker",
                "측정 작업이 완료되었습니다.",
                details={"report_path": str(report_path), "summary_path": str(summary_path), "profile_id": self.config.profile_id},
            )
            self.result_ready.emit(report, summary_text, str(report_path), str(summary_path), extra_paths)
        except Exception as exc:  # noqa: BLE001
            log_exception("measurement_worker", exc, details={"profile_id": self.config.profile_id, "port": self.config.port})
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class CameraExtractionWorker(QObject):
    progress_text = Signal(str)
    result_ready = Signal(str, str, str, float)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, video_path: Path, frame_csv_path: Path | None, output_dir: Path) -> None:
        super().__init__()
        self.video_path = video_path
        self.frame_csv_path = frame_csv_path
        self.output_dir = output_dir

    @Slot()
    def run(self) -> None:
        try:
            log_event("camera_extraction", "오프라인 카메라 추출을 시작합니다.", details={"video": str(self.video_path)})
            result = extract_camera_rppg_features(
                self.video_path,
                self.output_dir,
                frame_timestamps_path=self.frame_csv_path,
                status_callback=self.progress_text.emit,
            )
            self.result_ready.emit(
                str(result["features_csv_path"]),
                str(result["summary_json_path"]),
                str(result["selected_signal"]),
                float(result["selected_hr_bpm"]),
            )
        except Exception as exc:  # noqa: BLE001
            log_exception("camera_extraction", exc, details={"video": str(self.video_path)})
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("헬스케어 통합 스튜디오")
        configure_logging()
        self.store = ProductStore()
        self.resize(1080, 1920)
        self.setMinimumSize(900, 1400)

        self._thread: QThread | None = None
        self._worker: MeasurementWorker | None = None
        self._tool_thread: QThread | None = None
        self._camera_worker: CameraExtractionWorker | None = None
        self._camera_preview: cv2.VideoCapture | None = None
        self._preview_failures = 0
        self._last_report_path = ""
        self._last_summary_path = ""
        self._last_extra_paths: dict[str, str] = {}
        self._active_config: SessionConfig | None = None
        self._active_profile_id: int | None = None

        self.port_combo = QComboBox()
        self.camera_combo = QComboBox()
        self.baud_spin = QSpinBox()
        self.duration_spin = QDoubleSpinBox()
        self.sample_rate_spin = QDoubleSpinBox()
        self.age_input = QLineEdit()
        self.sex_combo = QComboBox()
        self.calibration_sbp_input = QLineEdit()
        self.calibration_dbp_input = QLineEdit()
        self.csv_input = QLineEdit()
        self.use_camera_checkbox = QCheckBox("카메라를 포함한 멀티모달 데이터셋 생성")
        self.camera_width_spin = QSpinBox()
        self.camera_height_spin = QSpinBox()
        self.camera_fps_spin = QDoubleSpinBox()
        self.camera_auto_exposure_checkbox = QCheckBox("자동 노출")
        self.camera_exposure_spin = QDoubleSpinBox()
        self.camera_auto_white_balance_checkbox = QCheckBox("자동 화이트밸런스")
        self.camera_white_balance_spin = QDoubleSpinBox()
        self.camera_gain_spin = QDoubleSpinBox()
        self.reconnect_checkbox = QCheckBox("장치 자동 재연결")
        self.serial_retry_spin = QSpinBox()
        self.camera_retry_spin = QSpinBox()
        self.no_data_timeout_spin = QDoubleSpinBox()
        self.camera_preview_label = QLabel("카메라 미리보기가 정지된 상태입니다.")
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("대기")
        self.summary_output = QPlainTextEdit()
        self.log_output = QPlainTextEdit()
        self.report_path_label = QLabel("-")
        self.summary_path_label = QLabel("-")
        self.analysis_mode_label = QLabel("-")
        self.overall_confidence_label = QLabel("-")
        self.no_read_outputs_label = QLabel("-")
        self.camera_video_label = QLabel("-")
        self.manifest_path_label = QLabel("-")
        self.camera_features_label = QLabel("-")
        self.camera_summary_label = QLabel("-")
        self.camera_perfusion_proxy_label = QLabel("-")
        self.camera_vascular_proxy_label = QLabel("-")
        self.oss_engine_label = QLabel("-")
        self.ml_model_label = QLabel("-")
        self.log_file_label = QLabel(str(LOG_PATH))
        self.active_profile_label = QLabel("-")
        self.video_input = QLineEdit()
        self.frame_csv_input = QLineEdit()
        self.extraction_output_dir_input = QLineEdit(str(OUTPUTS_DIR))
        self.camera_tool_output = QPlainTextEdit()
        self.camera_tool_features_label = QLabel("-")
        self.camera_tool_summary_label = QLabel("-")
        self.camera_tool_selected_signal_label = QLabel("-")
        self.camera_tool_selected_hr_label = QLabel("-")
        self.start_camera_extraction_button = QPushButton("영상에서 카메라 rPPG 추출")
        self.strategy_view = QPlainTextEdit()
        self.blueprint_view = QPlainTextEdit()
        self.profile_combo = QComboBox()
        self.profile_name_input = QLineEdit()
        self.profile_notes_input = QPlainTextEdit()
        self.profile_default_checkbox = QCheckBox("기본 프로필로 사용")
        self.profile_status_label = QLabel("-")
        self.session_history_output = QPlainTextEdit()
        self.diagnostics_output = QPlainTextEdit()
        self.save_profile_button = QPushButton("프로필 저장")
        self.new_profile_button = QPushButton("새 프로필")
        self.load_profile_button = QPushButton("프로필 불러오기")
        self.refresh_profiles_button = QPushButton("프로필 새로고침")
        self.refresh_history_button = QPushButton("이력 새로고침")
        self.start_live_button = QPushButton("라이브 측정 시작")
        self.analyze_csv_button = QPushButton("CSV 분석")
        self.refresh_ports_button = QPushButton("시리얼 포트 새로고침")
        self.refresh_cameras_button = QPushButton("카메라 새로고침")
        self.preview_camera_button = QPushButton("미리보기 시작")
        self.stop_preview_button = QPushButton("미리보기 중지")
        self.open_output_button = QPushButton("결과 폴더 열기")
        self.preview_timer = QTimer(self)

        self._build_ui()
        self._restore_ui_settings()
        self.refresh_ports()
        self.refresh_cameras()
        self.refresh_profiles()
        self.refresh_history_views()
        self._load_reference_docs()
        log_event("gui", "HealthCare UI를 시작했습니다.", details={"binding": QT_BINDING})

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel("헬스케어 통합 스튜디오")
        header.setStyleSheet("font-size: 26px; font-weight: 700;")
        subtitle = QLabel(
            "iPPG 측정, 카메라 동기 데이터셋 수집, 1.1 ~ 1.7 분석, 제품화 전략 확인을 한 화면에서 처리합니다."
        )
        subtitle.setStyleSheet("color: #5a6772;")

        tabs = QTabWidget()
        profile_tab = self._build_profiles_tab()
        tabs.addTab(self._build_measurement_tab(), "측정")
        tabs.addTab(self._build_camera_tools_tab(), "카메라 도구")
        tabs.addTab(self._build_strategy_tab(), "전략")

        tabs.insertTab(2, profile_tab, "프로필/이력")
        layout.addWidget(header)
        layout.addWidget(subtitle)
        layout.addWidget(tabs)
        self.setCentralWidget(central)

        refresh_action = QAction("시리얼 포트 새로고침", self)
        refresh_action.triggered.connect(self.refresh_ports)
        self.addAction(refresh_action)

        self.preview_timer.setInterval(50)
        self.preview_timer.timeout.connect(self.update_preview_frame)

    def _build_measurement_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        settings_group = QGroupBox("세션 설정")
        settings_layout = QGridLayout(settings_group)

        self.baud_spin.setRange(9600, 2_000_000)
        self.baud_spin.setValue(1_000_000)
        self.duration_spin.setRange(5.0, 600.0)
        self.duration_spin.setValue(60.0)
        self.duration_spin.setSuffix(" s")
        self.sample_rate_spin.setRange(20.0, 1000.0)
        self.sample_rate_spin.setValue(200.0)
        self.sample_rate_spin.setSuffix(" Hz")
        self.sex_combo.addItem("미입력", "unknown")
        self.sex_combo.addItem("남성", "male")
        self.sex_combo.addItem("여성", "female")
        self.csv_input.setPlaceholderText("오프라인 CSV 경로")
        self.age_input.setPlaceholderText("선택 입력")
        self.calibration_sbp_input.setPlaceholderText("선택 입력")
        self.calibration_dbp_input.setPlaceholderText("선택 입력")

        csv_browse_button = QToolButton()
        csv_browse_button.setText("찾기")
        csv_browse_button.clicked.connect(self.browse_csv)

        settings_layout.addWidget(QLabel("시리얼 포트"), 0, 0)
        settings_layout.addWidget(self.port_combo, 0, 1)
        settings_layout.addWidget(self.refresh_ports_button, 0, 2)
        settings_layout.addWidget(QLabel("활성 프로필"), 0, 3)
        settings_layout.addWidget(self.active_profile_label, 0, 4)
        settings_layout.addWidget(QLabel("보드레이트"), 1, 0)
        settings_layout.addWidget(self.baud_spin, 1, 1)
        settings_layout.addWidget(QLabel("측정 시간"), 1, 2)
        settings_layout.addWidget(self.duration_spin, 1, 3)
        settings_layout.addWidget(QLabel("샘플링 주파수"), 2, 0)
        settings_layout.addWidget(self.sample_rate_spin, 2, 1)
        settings_layout.addWidget(QLabel("나이"), 2, 2)
        settings_layout.addWidget(self.age_input, 2, 3)
        settings_layout.addWidget(QLabel("성별"), 3, 0)
        settings_layout.addWidget(self.sex_combo, 3, 1)
        settings_layout.addWidget(QLabel("보정 SBP"), 3, 2)
        settings_layout.addWidget(self.calibration_sbp_input, 3, 3)
        settings_layout.addWidget(QLabel("보정 DBP"), 4, 0)
        settings_layout.addWidget(self.calibration_dbp_input, 4, 1)
        settings_layout.addWidget(QLabel("CSV 입력"), 4, 2)
        csv_row = QHBoxLayout()
        csv_row.setContentsMargins(0, 0, 0, 0)
        csv_row.addWidget(self.csv_input)
        csv_row.addWidget(csv_browse_button)
        settings_layout.addLayout(csv_row, 4, 3)

        camera_group = QGroupBox("카메라 데이터셋")
        camera_layout = QGridLayout(camera_group)

        self.camera_width_spin.setRange(0, 7680)
        self.camera_width_spin.setValue(1920)
        self.camera_height_spin.setRange(0, 4320)
        self.camera_height_spin.setValue(1080)
        self.camera_fps_spin.setRange(0.0, 240.0)
        self.camera_fps_spin.setValue(30.0)
        self.camera_fps_spin.setSuffix(" fps")
        self.camera_auto_exposure_checkbox.setChecked(True)
        self.camera_exposure_spin.setRange(-13.0, 13.0)
        self.camera_exposure_spin.setDecimals(2)
        self.camera_exposure_spin.setValue(-6.0)
        self.camera_auto_white_balance_checkbox.setChecked(True)
        self.camera_white_balance_spin.setRange(2000.0, 9000.0)
        self.camera_white_balance_spin.setValue(4500.0)
        self.camera_white_balance_spin.setSuffix(" K")
        self.camera_gain_spin.setRange(0.0, 128.0)
        self.camera_gain_spin.setDecimals(2)
        self.camera_gain_spin.setValue(0.0)
        self.reconnect_checkbox.setChecked(True)
        self.serial_retry_spin.setRange(0, 10)
        self.serial_retry_spin.setValue(2)
        self.camera_retry_spin.setRange(0, 10)
        self.camera_retry_spin.setValue(2)
        self.no_data_timeout_spin.setRange(1.0, 60.0)
        self.no_data_timeout_spin.setDecimals(1)
        self.no_data_timeout_spin.setValue(5.0)
        self.no_data_timeout_spin.setSuffix(" s")

        camera_layout.addWidget(self.use_camera_checkbox, 0, 0, 1, 2)
        camera_layout.addWidget(QLabel("카메라"), 1, 0)
        camera_layout.addWidget(self.camera_combo, 1, 1)
        camera_layout.addWidget(self.refresh_cameras_button, 1, 2)
        camera_layout.addWidget(QLabel("너비"), 2, 0)
        camera_layout.addWidget(self.camera_width_spin, 2, 1)
        camera_layout.addWidget(QLabel("높이"), 2, 2)
        camera_layout.addWidget(self.camera_height_spin, 2, 3)
        camera_layout.addWidget(QLabel("FPS"), 3, 0)
        camera_layout.addWidget(self.camera_fps_spin, 3, 1)
        camera_layout.addWidget(self.preview_camera_button, 3, 2)
        camera_layout.addWidget(self.stop_preview_button, 3, 3)
        camera_layout.addWidget(self.camera_auto_exposure_checkbox, 4, 0)
        camera_layout.addWidget(self.camera_exposure_spin, 4, 1)
        camera_layout.addWidget(self.camera_auto_white_balance_checkbox, 4, 2)
        camera_layout.addWidget(self.camera_white_balance_spin, 4, 3)
        camera_layout.addWidget(QLabel("Gain"), 5, 0)
        camera_layout.addWidget(self.camera_gain_spin, 5, 1)
        camera_layout.addWidget(self.reconnect_checkbox, 5, 2, 1, 2)
        camera_layout.addWidget(QLabel("시리얼 재시도"), 6, 0)
        camera_layout.addWidget(self.serial_retry_spin, 6, 1)
        camera_layout.addWidget(QLabel("카메라 재시도"), 6, 2)
        camera_layout.addWidget(self.camera_retry_spin, 6, 3)
        camera_layout.addWidget(QLabel("무데이터 타임아웃"), 7, 0)
        camera_layout.addWidget(self.no_data_timeout_spin, 7, 1)

        preview_group = QGroupBox("미리보기")
        preview_layout = QVBoxLayout(preview_group)
        self.camera_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_preview_label.setMinimumHeight(280)
        self.camera_preview_label.setStyleSheet("background: #10161a; color: #c7d0d9; border: 1px solid #23303b;")
        preview_layout.addWidget(self.camera_preview_label)

        actions_row = QHBoxLayout()
        actions_row.addWidget(self.start_live_button)
        actions_row.addWidget(self.analyze_csv_button)
        actions_row.addWidget(self.open_output_button)
        actions_row.addStretch(1)

        status_group = QGroupBox("실행 상태")
        status_layout = QFormLayout(status_group)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        status_layout.addRow("상태", self.status_label)
        status_layout.addRow("진행", self.progress_bar)
        status_layout.addRow("분석 모드", self.analysis_mode_label)
        status_layout.addRow("전체 신뢰도", self.overall_confidence_label)
        status_layout.addRow("무응답 권장", self.no_read_outputs_label)
        status_layout.addRow("리포트 JSON", self.report_path_label)
        status_layout.addRow("요약 TXT", self.summary_path_label)
        status_layout.addRow("카메라 영상", self.camera_video_label)
        status_layout.addRow("매니페스트", self.manifest_path_label)
        status_layout.addRow("카메라 특징", self.camera_features_label)
        status_layout.addRow("카메라 요약", self.camera_summary_label)
        status_layout.addRow("카메라 관류 프록시", self.camera_perfusion_proxy_label)
        status_layout.addRow("카메라 혈관 프록시", self.camera_vascular_proxy_label)
        status_layout.addRow("오픈소스 엔진", self.oss_engine_label)
        status_layout.addRow("ML 모델", self.ml_model_label)
        status_layout.addRow("로그 파일", self.log_file_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlaceholderText("분석 요약이 여기에 표시됩니다.")
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("실행 로그가 여기에 표시됩니다.")
        self.summary_output.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self.log_output.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        splitter.addWidget(self._wrap_panel("분석 요약", self.summary_output))
        splitter.addWidget(self._wrap_panel("실행 로그", self.log_output))
        splitter.setSizes([800, 500])

        layout.addWidget(settings_group)
        layout.addWidget(camera_group)
        layout.addWidget(preview_group)
        layout.addLayout(actions_row)
        layout.addWidget(status_group)
        layout.addWidget(splitter, 1)

        self.refresh_ports_button.clicked.connect(self.refresh_ports)
        self.refresh_cameras_button.clicked.connect(self.refresh_cameras)
        self.preview_camera_button.clicked.connect(self.start_preview)
        self.stop_preview_button.clicked.connect(self.stop_preview)
        self.start_live_button.clicked.connect(self.start_live_measurement)
        self.analyze_csv_button.clicked.connect(self.start_csv_analysis)
        self.open_output_button.clicked.connect(self.open_last_output_dir)

        return container

    def _build_camera_tools_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        tool_group = QGroupBox("오프라인 카메라 rPPG 추출")
        tool_layout = QGridLayout(tool_group)

        self.video_input.setPlaceholderText("카메라 영상 파일 경로")
        self.frame_csv_input.setPlaceholderText("선택: 카메라 프레임 CSV 경로")
        self.extraction_output_dir_input.setPlaceholderText("출력 디렉토리")

        browse_video_button = QToolButton()
        browse_video_button.setText("영상")
        browse_video_button.clicked.connect(self.browse_video)

        browse_frame_csv_button = QToolButton()
        browse_frame_csv_button.setText("프레임 CSV")
        browse_frame_csv_button.clicked.connect(self.browse_frame_csv)

        browse_output_dir_button = QToolButton()
        browse_output_dir_button.setText("출력")
        browse_output_dir_button.clicked.connect(self.browse_extraction_output_dir)

        tool_layout.addWidget(QLabel("영상"), 0, 0)
        tool_layout.addWidget(self.video_input, 0, 1)
        tool_layout.addWidget(browse_video_button, 0, 2)
        tool_layout.addWidget(QLabel("프레임 CSV"), 1, 0)
        tool_layout.addWidget(self.frame_csv_input, 1, 1)
        tool_layout.addWidget(browse_frame_csv_button, 1, 2)
        tool_layout.addWidget(QLabel("출력 폴더"), 2, 0)
        tool_layout.addWidget(self.extraction_output_dir_input, 2, 1)
        tool_layout.addWidget(browse_output_dir_button, 2, 2)
        tool_layout.addWidget(self.start_camera_extraction_button, 3, 0, 1, 3)

        result_group = QGroupBox("카메라 추출 결과")
        result_layout = QFormLayout(result_group)
        result_layout.addRow("특징 CSV", self.camera_tool_features_label)
        result_layout.addRow("요약 JSON", self.camera_tool_summary_label)
        result_layout.addRow("선택 신호", self.camera_tool_selected_signal_label)
        result_layout.addRow("추정 HR", self.camera_tool_selected_hr_label)

        self.camera_tool_output.setReadOnly(True)
        self.camera_tool_output.setPlaceholderText("카메라 추출 로그가 여기에 표시됩니다.")
        self.camera_tool_output.setWordWrapMode(QTextOption.WrapMode.NoWrap)

        layout.addWidget(tool_group)
        layout.addWidget(result_group)
        layout.addWidget(self._wrap_panel("카메라 추출 로그", self.camera_tool_output), 1)

        self.start_camera_extraction_button.clicked.connect(self.start_camera_extraction)
        return container

    def _build_profiles_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        profile_group = QGroupBox("프로필 관리")
        profile_layout = QGridLayout(profile_group)
        self.profile_notes_input.setMaximumBlockCount(200)

        profile_layout.addWidget(QLabel("저장된 프로필"), 0, 0)
        profile_layout.addWidget(self.profile_combo, 0, 1)
        profile_layout.addWidget(self.load_profile_button, 0, 2)
        profile_layout.addWidget(self.refresh_profiles_button, 0, 3)
        profile_layout.addWidget(QLabel("프로필 이름"), 1, 0)
        profile_layout.addWidget(self.profile_name_input, 1, 1)
        profile_layout.addWidget(self.profile_default_checkbox, 1, 2, 1, 2)
        profile_layout.addWidget(QLabel("메모"), 2, 0)
        profile_layout.addWidget(self.profile_notes_input, 2, 1, 2, 3)
        profile_layout.addWidget(self.save_profile_button, 4, 1)
        profile_layout.addWidget(self.new_profile_button, 4, 2)
        profile_layout.addWidget(QLabel("상태"), 5, 0)
        profile_layout.addWidget(self.profile_status_label, 5, 1, 1, 3)

        tools_row = QHBoxLayout()
        tools_row.addWidget(self.refresh_history_button)
        tools_row.addStretch(1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.session_history_output.setReadOnly(True)
        self.diagnostics_output.setReadOnly(True)
        self.session_history_output.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self.diagnostics_output.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        splitter.addWidget(self._wrap_panel("최근 측정 이력", self.session_history_output))
        splitter.addWidget(self._wrap_panel("최근 진단 로그", self.diagnostics_output))
        splitter.setSizes([700, 500])

        layout.addWidget(profile_group)
        layout.addLayout(tools_row)
        layout.addWidget(splitter, 1)

        self.load_profile_button.clicked.connect(self.load_selected_profile)
        self.save_profile_button.clicked.connect(self.save_profile)
        self.new_profile_button.clicked.connect(self.new_profile)
        self.refresh_profiles_button.clicked.connect(self.refresh_profiles)
        self.refresh_history_button.clicked.connect(self.refresh_history_views)
        return container

    def _build_strategy_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        strategy_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.strategy_view.setReadOnly(True)
        self.blueprint_view.setReadOnly(True)
        self.strategy_view.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self.blueprint_view.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        strategy_splitter.addWidget(self._wrap_panel("짧은 요약", self.strategy_view))
        strategy_splitter.addWidget(self._wrap_panel("상세 청사진", self.blueprint_view))
        strategy_splitter.setSizes([520, 800])
        layout.addWidget(strategy_splitter)
        return container

    def _wrap_panel(self, title: str, widget: QWidget) -> QWidget:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _load_reference_docs(self) -> None:
        self.strategy_view.setPlainText(self._read_text(SUMMARY_MD_PATH))
        self.blueprint_view.setPlainText(self._read_text(BLUEPRINT_MD_PATH))

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"문서를 읽을 수 없습니다: {path}\n{exc}"

    def _restore_ui_settings(self) -> None:
        settings = self.store.load_settings("ui_state")
        if not settings:
            return
        self.baud_spin.setValue(int(settings.get("baud", 1_000_000)))
        self.duration_spin.setValue(float(settings.get("duration_s", 60.0)))
        self.sample_rate_spin.setValue(float(settings.get("sample_rate_hz", 200.0)))
        self.age_input.setText(str(settings.get("age", "") or ""))
        self.calibration_sbp_input.setText(str(settings.get("calibration_sbp", "") or ""))
        self.calibration_dbp_input.setText(str(settings.get("calibration_dbp", "") or ""))
        self.camera_width_spin.setValue(int(settings.get("camera_width", 1920)))
        self.camera_height_spin.setValue(int(settings.get("camera_height", 1080)))
        self.camera_fps_spin.setValue(float(settings.get("camera_fps", 30.0)))
        self.camera_auto_exposure_checkbox.setChecked(bool(settings.get("camera_auto_exposure", True)))
        self.camera_exposure_spin.setValue(float(settings.get("camera_exposure", -6.0)))
        self.camera_auto_white_balance_checkbox.setChecked(bool(settings.get("camera_auto_white_balance", True)))
        self.camera_white_balance_spin.setValue(float(settings.get("camera_white_balance", 4500.0)))
        self.camera_gain_spin.setValue(float(settings.get("camera_gain", 0.0)))
        self.reconnect_checkbox.setChecked(bool(settings.get("reconnect_enabled", True)))
        self.serial_retry_spin.setValue(int(settings.get("serial_retry_count", 2)))
        self.camera_retry_spin.setValue(int(settings.get("camera_retry_count", 2)))
        self.no_data_timeout_spin.setValue(float(settings.get("no_data_timeout_s", 5.0)))
        self._active_profile_id = try_int(str(settings.get("profile_id", "") or ""))

    def _save_ui_settings(self) -> None:
        self.store.save_settings(
            "ui_state",
            {
                "baud": int(self.baud_spin.value()),
                "duration_s": float(self.duration_spin.value()),
                "sample_rate_hz": float(self.sample_rate_spin.value()),
                "age": self.age_input.text().strip(),
                "calibration_sbp": self.calibration_sbp_input.text().strip(),
                "calibration_dbp": self.calibration_dbp_input.text().strip(),
                "camera_width": int(self.camera_width_spin.value()),
                "camera_height": int(self.camera_height_spin.value()),
                "camera_fps": float(self.camera_fps_spin.value()),
                "camera_auto_exposure": bool(self.camera_auto_exposure_checkbox.isChecked()),
                "camera_exposure": float(self.camera_exposure_spin.value()),
                "camera_auto_white_balance": bool(self.camera_auto_white_balance_checkbox.isChecked()),
                "camera_white_balance": float(self.camera_white_balance_spin.value()),
                "camera_gain": float(self.camera_gain_spin.value()),
                "reconnect_enabled": bool(self.reconnect_checkbox.isChecked()),
                "serial_retry_count": int(self.serial_retry_spin.value()),
                "camera_retry_count": int(self.camera_retry_spin.value()),
                "no_data_timeout_s": float(self.no_data_timeout_spin.value()),
                "profile_id": self._active_profile_id,
            },
        )

    def refresh_profiles(self) -> None:
        profiles = self.store.list_profiles()
        selected_profile_id = self._active_profile_id
        self.profile_combo.clear()
        for profile in profiles:
            label = f"{profile['name']} | age {profile.get('age') or '-'} | {profile.get('sex') or 'unknown'}"
            if profile.get("is_default"):
                label += " | 기본"
            self.profile_combo.addItem(label, int(profile["id"]))
        if selected_profile_id is not None:
            index = self.profile_combo.findData(selected_profile_id)
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)
                profile = self.store.get_profile(int(selected_profile_id))
                if profile:
                    self.apply_profile(profile)
        elif profiles:
            default_profile = next((item for item in profiles if item.get("is_default")), profiles[0])
            self.apply_profile(default_profile)

    def refresh_history_views(self) -> None:
        sessions = self.store.list_sessions(limit=40)
        diagnostics = self.store.list_diagnostics(limit=60)
        session_lines: list[str] = []
        for session in sessions:
            no_read = ", ".join(session.get("no_read_outputs") or []) or "없음"
            session_lines.append(
                " | ".join(
                    [
                        session.get("created_at") or "-",
                        f"상태 {session.get('status') or '-'}",
                        f"프로필 {session.get('profile_name') or '-'}",
                        f"모드 {session.get('analysis_mode_label') or session.get('mode') or '-'}",
                        f"HR {float(session.get('heart_rate_bpm') or 0.0):.1f}",
                        f"신뢰도 {float(session.get('overall_confidence_score') or 0.0):.1f}",
                        f"무응답 {no_read}",
                    ]
                )
            )
        self.session_history_output.setPlainText("\n".join(session_lines) if session_lines else "측정 이력이 없습니다.")
        diagnostic_lines = [
            f"{item.get('created_at')} | {item.get('level')} | {item.get('source')} | {item.get('message')} | {item.get('details_json') or ''}"
            for item in diagnostics
        ]
        self.diagnostics_output.setPlainText("\n".join(diagnostic_lines) if diagnostic_lines else "진단 로그가 없습니다.")

    def refresh_ports(self) -> None:
        ports = list_serial_port_rows()
        current = self.port_combo.currentData()
        self.port_combo.clear()
        for port in ports:
            self.port_combo.addItem(describe_serial_port(port), port.device)
        if current:
            index = self.port_combo.findData(current)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)
        if self.port_combo.count() == 0:
            self.port_combo.addItem("감지된 시리얼 포트 없음", None)

    def refresh_cameras(self) -> None:
        cameras = probe_camera_indices(max_index=5)
        current = self.camera_combo.currentData()
        self.camera_combo.clear()
        for camera_info in cameras:
            self.camera_combo.addItem(describe_camera_entry(camera_info), int(camera_info["index"]))
        if current is not None:
            index = self.camera_combo.findData(current)
            if index >= 0:
                self.camera_combo.setCurrentIndex(index)
        if self.camera_combo.count() == 0:
            self.camera_combo.addItem("감지된 카메라 없음", None)

    def browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "CSV 입력 파일 선택",
            str(OUTPUTS_DIR),
            "CSV 파일 (*.csv);;모든 파일 (*)",
        )
        if path:
            self.csv_input.setText(path)

    def browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "카메라 영상 선택",
            str(OUTPUTS_DIR),
            "동영상 파일 (*.mp4 *.avi *.mov);;모든 파일 (*)",
        )
        if path:
            self.video_input.setText(path)

    def browse_frame_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "카메라 프레임 CSV 선택",
            str(OUTPUTS_DIR),
            "CSV 파일 (*.csv);;모든 파일 (*)",
        )
        if path:
            self.frame_csv_input.setText(path)

    def browse_extraction_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "출력 폴더 선택", self.extraction_output_dir_input.text().strip() or str(OUTPUTS_DIR))
        if path:
            self.extraction_output_dir_input.setText(path)

    def _camera_open_kwargs(self) -> dict[str, Any]:
        return {
            "width": int(self.camera_width_spin.value()) or None,
            "height": int(self.camera_height_spin.value()) or None,
            "fps": float(self.camera_fps_spin.value()) or None,
            "auto_exposure": bool(self.camera_auto_exposure_checkbox.isChecked()),
            "exposure_value": None if self.camera_auto_exposure_checkbox.isChecked() else float(self.camera_exposure_spin.value()),
            "auto_white_balance": bool(self.camera_auto_white_balance_checkbox.isChecked()),
            "white_balance_value": None if self.camera_auto_white_balance_checkbox.isChecked() else float(self.camera_white_balance_spin.value()),
            "gain_value": float(self.camera_gain_spin.value()),
        }

    def start_preview(self) -> None:
        camera_index = self.camera_combo.currentData()
        if camera_index is None:
            QMessageBox.information(self, "카메라 없음", "현재 사용 가능한 카메라가 없습니다.")
            return

        self.stop_preview()
        self._camera_preview = open_camera_capture(int(camera_index), **self._camera_open_kwargs())
        if self._camera_preview is None:
            QMessageBox.critical(self, "미리보기 실패", f"카메라 {camera_index} 을(를) 열 수 없습니다.")
            return

        self._preview_failures = 0
        self.preview_timer.start()
        self.append_log(f"카메라 {camera_index} 미리보기를 시작했습니다.")

    def stop_preview(self) -> None:
        self.preview_timer.stop()
        if self._camera_preview is not None:
            self._camera_preview.release()
            self._camera_preview = None
        self._preview_failures = 0
        self.camera_preview_label.setPixmap(QPixmap())
        self.camera_preview_label.setText("카메라 미리보기가 정지된 상태입니다.")

    @Slot()
    def update_preview_frame(self) -> None:
        if self._camera_preview is None:
            return

        ok, frame = self._camera_preview.read()
        if not ok or frame is None:
            self._preview_failures += 1
            if self.reconnect_checkbox.isChecked() and self._preview_failures >= 8:
                camera_index = self.camera_combo.currentData()
                if self._camera_preview is not None:
                    self._camera_preview.release()
                self._camera_preview = open_camera_capture(int(camera_index), **self._camera_open_kwargs()) if camera_index is not None else None
                self._preview_failures = 0
            return

        self._preview_failures = 0
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb_frame.shape[:2]
        image = QImage(rgb_frame.data, width, height, rgb_frame.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.camera_preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_preview_label.setPixmap(pixmap)

    def start_live_measurement(self) -> None:
        try:
            config = self._build_config(csv_mode=False)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "설정 오류", str(exc))
            return
        self._start_worker(config)

    def start_csv_analysis(self) -> None:
        try:
            config = self._build_config(csv_mode=True)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "설정 오류", str(exc))
            return
        self._start_worker(config)

    def start_camera_extraction(self) -> None:
        if self._tool_thread is not None:
            QMessageBox.information(self, "작업 중", "카메라 추출 작업이 이미 실행 중입니다.")
            return

        video_text = self.video_input.text().strip()
        if not video_text:
            QMessageBox.critical(self, "설정 오류", "추출을 시작하기 전에 영상 파일을 선택하세요.")
            return

        video_path = Path(video_text).resolve()
        frame_csv_text = self.frame_csv_input.text().strip()
        frame_csv_path = Path(frame_csv_text).resolve() if frame_csv_text else None
        output_dir_text = self.extraction_output_dir_input.text().strip()
        output_dir = Path(output_dir_text).resolve() if output_dir_text else video_path.parent

        self.camera_tool_output.clear()
        self.camera_tool_features_label.setText("-")
        self.camera_tool_summary_label.setText("-")
        self.camera_tool_selected_signal_label.setText("-")
        self.camera_tool_selected_hr_label.setText("-")
        self.start_camera_extraction_button.setEnabled(False)

        self._tool_thread = QThread(self)
        self._camera_worker = CameraExtractionWorker(video_path, frame_csv_path, output_dir)
        self._camera_worker.moveToThread(self._tool_thread)
        self._tool_thread.started.connect(self._camera_worker.run)
        self._camera_worker.progress_text.connect(self.append_camera_tool_log)
        self._camera_worker.result_ready.connect(self.handle_camera_extraction_result)
        self._camera_worker.failed.connect(self.handle_camera_extraction_failure)
        self._camera_worker.finished.connect(self._tool_thread.quit)
        self._camera_worker.finished.connect(self._camera_worker.deleteLater)
        self._tool_thread.finished.connect(self._tool_thread.deleteLater)
        self._tool_thread.finished.connect(self._cleanup_camera_extraction)
        self._tool_thread.start()

    def _build_config(self, csv_mode: bool) -> SessionConfig:
        csv_path_text = self.csv_input.text().strip()
        csv_path = Path(csv_path_text).resolve() if csv_mode and csv_path_text else None
        if csv_mode and csv_path is None:
            raise ValueError("CSV 분석을 시작하기 전에 CSV 파일을 선택하세요.")

        port = self.port_combo.currentData()
        age_text = self.age_input.text().strip()
        calibration_sbp_text = self.calibration_sbp_input.text().strip()
        calibration_dbp_text = self.calibration_dbp_input.text().strip()
        camera_index = self.camera_combo.currentData()

        return SessionConfig(
            port=str(port) if port else None,
            baud=int(self.baud_spin.value()),
            duration_s=float(self.duration_spin.value()),
            sample_rate_hz=float(self.sample_rate_spin.value()),
            age=int(age_text) if age_text else None,
            sex=str(self.sex_combo.currentData() or self.sex_combo.currentText()),
            calibration_sbp=float(calibration_sbp_text) if calibration_sbp_text else None,
            calibration_dbp=float(calibration_dbp_text) if calibration_dbp_text else None,
            csv_input=csv_path,
            use_camera_dataset=bool(self.use_camera_checkbox.isChecked()) and not csv_mode,
            camera_index=int(camera_index) if camera_index is not None else None,
            camera_width=int(self.camera_width_spin.value()) or None,
            camera_height=int(self.camera_height_spin.value()) or None,
            camera_fps=float(self.camera_fps_spin.value()) or None,
            camera_auto_exposure=bool(self.camera_auto_exposure_checkbox.isChecked()),
            camera_exposure_value=None if self.camera_auto_exposure_checkbox.isChecked() else float(self.camera_exposure_spin.value()),
            camera_auto_white_balance=bool(self.camera_auto_white_balance_checkbox.isChecked()),
            camera_white_balance_value=None if self.camera_auto_white_balance_checkbox.isChecked() else float(self.camera_white_balance_spin.value()),
            camera_gain_value=float(self.camera_gain_spin.value()),
            serial_retry_count=int(self.serial_retry_spin.value()),
            camera_retry_count=int(self.camera_retry_spin.value()),
            reconnect_enabled=bool(self.reconnect_checkbox.isChecked()),
            no_data_timeout_s=float(self.no_data_timeout_spin.value()),
            profile_id=self._active_profile_id,
            profile_name=self.profile_name_input.text().strip(),
            profile_notes=self.profile_notes_input.toPlainText().strip(),
        )

    def _start_worker(self, config: SessionConfig) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "작업 중", "측정 또는 분석 작업이 이미 실행 중입니다.")
            return

        self._active_config = config
        self._save_ui_settings()
        if config.profile_id is not None:
            self.store.mark_profile_used(config.profile_id)
            self.refresh_profiles()

        self.summary_output.clear()
        self.log_output.clear()
        self.status_label.setText("실행 중")
        self.progress_bar.setRange(0, 0)
        self.report_path_label.setText("-")
        self.summary_path_label.setText("-")
        self.analysis_mode_label.setText("-")
        self.overall_confidence_label.setText("-")
        self.no_read_outputs_label.setText("-")
        self.camera_video_label.setText("-")
        self.manifest_path_label.setText("-")
        self.camera_features_label.setText("-")
        self.camera_summary_label.setText("-")
        self.camera_perfusion_proxy_label.setText("-")
        self.camera_vascular_proxy_label.setText("-")
        self.oss_engine_label.setText("-")
        self.ml_model_label.setText("-")
        self.log_file_label.setText(str(LOG_PATH))
        self._last_extra_paths = {}

        self.start_live_button.setEnabled(False)
        self.analyze_csv_button.setEnabled(False)

        self._thread = QThread(self)
        self._worker = MeasurementWorker(config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress_text.connect(self.append_log)
        self._worker.result_ready.connect(self.handle_result)
        self._worker.failed.connect(self.handle_failure)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._cleanup_after_run)
        self._thread.start()

    @Slot()
    def _cleanup_after_run(self) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1 if self.status_label.text() == "완료" else 0)
        self.start_live_button.setEnabled(True)
        self.analyze_csv_button.setEnabled(True)
        self._thread = None
        self._worker = None

    @Slot()
    def _cleanup_camera_extraction(self) -> None:
        self.start_camera_extraction_button.setEnabled(True)
        self._tool_thread = None
        self._camera_worker = None

    @Slot(str)
    def append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    @Slot(str)
    def append_camera_tool_log(self, message: str) -> None:
        self.camera_tool_output.appendPlainText(message)

    @Slot(dict, str, str, str, dict)
    def handle_result(
        self,
        report: dict[str, Any],
        summary_text: str,
        report_path: str,
        summary_path: str,
        extra_paths: dict[str, str],
    ) -> None:
        self.status_label.setText("완료")
        self.summary_output.setPlainText(summary_text)
        self.report_path_label.setText(report_path)
        self.summary_path_label.setText(summary_path)
        metadata = report.get("metadata") or {}
        self.analysis_mode_label.setText(str(metadata.get("measurement_mode_label") or "-"))
        quality_report = report.get("quality") or {}
        self.overall_confidence_label.setText(f"{float(quality_report.get('overall_confidence_score') or 0.0):.1f}")
        no_read_outputs = quality_report.get("no_read_outputs") or []
        self.no_read_outputs_label.setText(", ".join(str(item) for item in no_read_outputs) if no_read_outputs else "없음")
        self._last_report_path = report_path
        self._last_summary_path = summary_path
        self._last_extra_paths = extra_paths
        open_source_report = report.get("open_source") or {}
        if open_source_report.get("available"):
            self.oss_engine_label.setText(str(open_source_report.get("engine_label") or "-"))
        else:
            self.oss_engine_label.setText("미적용")
        ml_report = report.get("ml") or {}
        if ml_report.get("available"):
            bundle_kind = "기본 번들" if ml_report.get("bootstrap_bundle") else "사용자 학습 번들"
            self.ml_model_label.setText(f"{ml_report.get('bundle_version') or '-'} | {bundle_kind}")
        else:
            self.ml_model_label.setText("미적용")
        self.log_file_label.setText(extra_paths.get("log_path") or str(LOG_PATH))

        if extra_paths.get("camera_video"):
            self.camera_video_label.setText(extra_paths["camera_video"])
        if extra_paths.get("session_manifest"):
            self.manifest_path_label.setText(extra_paths["session_manifest"])
        if extra_paths.get("camera_features_csv"):
            self.camera_features_label.setText(extra_paths["camera_features_csv"])
        if extra_paths.get("camera_summary_json"):
            self.camera_summary_label.setText(extra_paths["camera_summary_json"])
        camera_report = report.get("camera") or {}
        if camera_report.get("available"):
            self.camera_perfusion_proxy_label.setText(f"{float(camera_report.get('camera_perfusion_proxy_score') or 0.0):.1f}")
            self.camera_vascular_proxy_label.setText(f"{float(camera_report.get('camera_vascular_proxy_score') or 0.0):.1f}")

        warnings = report.get("warnings") or []
        if warnings:
            self.log_output.appendPlainText("")
            self.log_output.appendPlainText("경고:")
            for warning in warnings:
                self.log_output.appendPlainText(f"- {warning}")

        heart_rate = report.get("heart_rate") or {}
        session_id = self.store.record_session(
            profile_id=self._active_config.profile_id if self._active_config else None,
            status="completed",
            mode="csv" if self._active_config and self._active_config.csv_input else "multimodal" if self._active_config and self._active_config.use_camera_dataset else "live",
            port=self._active_config.port if self._active_config else None,
            duration_s=self._active_config.duration_s if self._active_config else None,
            output_dir=str(Path(report_path).resolve().parent),
            report_path=report_path,
            summary_path=summary_path,
            signal_quality_score=float(metadata.get("signal_quality_score") or 0.0),
            overall_confidence_score=float(quality_report.get("overall_confidence_score") or 0.0),
            heart_rate_bpm=float(heart_rate.get("heart_rate_bpm") or 0.0),
            analysis_mode_label=str(metadata.get("measurement_mode_label") or "-"),
            no_read_outputs=list(no_read_outputs),
        )
        for warning in warnings:
            self.store.record_diagnostic(level="warning", source="analysis", message=str(warning), details={"report_path": report_path}, session_id=session_id)
        self.refresh_history_views()

    @Slot(str)
    def handle_failure(self, message: str) -> None:
        self.status_label.setText("실패")
        self.log_output.appendPlainText("")
        self.log_output.appendPlainText(f"오류: {message}")
        session_id = self.store.record_session(
            profile_id=self._active_config.profile_id if self._active_config else None,
            status="failed",
            mode="csv" if self._active_config and self._active_config.csv_input else "multimodal" if self._active_config and self._active_config.use_camera_dataset else "live",
            port=self._active_config.port if self._active_config else None,
            duration_s=self._active_config.duration_s if self._active_config else None,
            output_dir=self._last_extra_paths.get("output_dir"),
            report_path=None,
            summary_path=None,
            signal_quality_score=None,
            overall_confidence_score=None,
            heart_rate_bpm=None,
            analysis_mode_label=None,
            no_read_outputs=[],
        )
        self.store.record_diagnostic(
            level="error",
            source="ui",
            message=str(message),
            details={"profile_id": self._active_config.profile_id if self._active_config else None},
            session_id=session_id,
        )
        self.refresh_history_views()
        QMessageBox.critical(self, "측정 실패", message)

    @Slot(str, str, str, float)
    def handle_camera_extraction_result(
        self,
        features_csv_path: str,
        summary_json_path: str,
        selected_signal: str,
        selected_hr_bpm: float,
    ) -> None:
        self.camera_tool_features_label.setText(features_csv_path)
        self.camera_tool_summary_label.setText(summary_json_path)
        self.camera_tool_selected_signal_label.setText(translate_camera_signal_name(selected_signal))
        self.camera_tool_selected_hr_label.setText(f"{selected_hr_bpm:.2f} bpm")
        self.camera_tool_output.appendPlainText("")
        self.camera_tool_output.appendPlainText("카메라 추출이 완료되었습니다.")

    @Slot(str)
    def handle_camera_extraction_failure(self, message: str) -> None:
        self.camera_tool_output.appendPlainText("")
        self.camera_tool_output.appendPlainText(f"오류: {message}")
        QMessageBox.critical(self, "카메라 추출 실패", message)

    def apply_profile(self, profile: dict[str, Any]) -> None:
        self._active_profile_id = int(profile["id"])
        self.profile_name_input.setText(str(profile.get("name") or ""))
        self.profile_notes_input.setPlainText(str(profile.get("notes") or ""))
        self.profile_default_checkbox.setChecked(bool(profile.get("is_default")))
        self.age_input.setText("" if profile.get("age") is None else str(profile.get("age")))
        self.calibration_sbp_input.setText("" if profile.get("calibration_sbp") is None else str(profile.get("calibration_sbp")))
        self.calibration_dbp_input.setText("" if profile.get("calibration_dbp") is None else str(profile.get("calibration_dbp")))
        sex_value = str(profile.get("sex") or "unknown")
        sex_index = self.sex_combo.findData(sex_value)
        if sex_index >= 0:
            self.sex_combo.setCurrentIndex(sex_index)
        self.active_profile_label.setText(f"{profile.get('name')} (ID {profile.get('id')})")
        self.profile_status_label.setText(f"프로필 {profile.get('name')} 적용 완료")
        self._save_ui_settings()

    def load_selected_profile(self) -> None:
        profile_id = self.profile_combo.currentData()
        if profile_id is None:
            QMessageBox.information(self, "프로필 없음", "불러올 프로필이 없습니다.")
            return
        profile = self.store.get_profile(int(profile_id))
        if not profile:
            QMessageBox.critical(self, "프로필 오류", "선택한 프로필을 찾을 수 없습니다.")
            return
        self.apply_profile(profile)

    def save_profile(self) -> None:
        name = self.profile_name_input.text().strip()
        if not name:
            QMessageBox.critical(self, "프로필 오류", "프로필 이름을 입력하세요.")
            return
        profile = self.store.upsert_profile(
            name=name,
            age=try_int(self.age_input.text() or ""),
            sex=str(self.sex_combo.currentData() or self.sex_combo.currentText()),
            calibration_sbp=try_float(self.calibration_sbp_input.text() or ""),
            calibration_dbp=try_float(self.calibration_dbp_input.text() or ""),
            notes=self.profile_notes_input.toPlainText().strip(),
            profile_id=self._active_profile_id,
            make_default=bool(self.profile_default_checkbox.isChecked()),
        )
        self.apply_profile(profile)
        self.refresh_profiles()
        log_event("profiles", "프로필을 저장했습니다.", details={"profile_id": int(profile["id"]), "name": profile.get("name")})

    def new_profile(self) -> None:
        self._active_profile_id = None
        self.profile_name_input.clear()
        self.profile_notes_input.clear()
        self.profile_default_checkbox.setChecked(False)
        self.active_profile_label.setText("새 프로필")
        self.profile_status_label.setText("새 프로필 입력 상태")

    def open_last_output_dir(self) -> None:
        target_path = ""
        if self._last_report_path:
            target_path = str(Path(self._last_report_path).resolve().parent)
        elif OUTPUTS_DIR.exists():
            target_path = str(OUTPUTS_DIR.resolve())

        if not target_path:
            QMessageBox.information(self, "결과 없음", "아직 열 수 있는 결과 폴더가 없습니다.")
            return

        if sys.platform.startswith("win"):
            os.startfile(target_path)  # type: ignore[attr-defined]
        else:
            QMessageBox.information(self, "결과 폴더", target_path)

    def closeEvent(self, event: Any) -> None:
        self.stop_preview()
        self._save_ui_settings()
        log_event("gui", "HealthCare UI를 종료합니다.")
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
