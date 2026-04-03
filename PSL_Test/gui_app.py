from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from analysis_pipeline import build_user_profile, format_summary_text, load_camera_summary, run_analysis, write_report_files
from app_paths import LOG_PATH, new_session_dir
from camera_rppg import capture_multimodal_session, extract_camera_rppg_features, open_camera_capture, probe_camera_indices
from serial_capture import capture_serial_session, list_serial_ports, load_dataset_from_csv, write_capture_csv

try:
    from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    def image_format_rgb888():
        return QImage.Format.Format_RGB888

except ImportError:
    from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    def image_format_rgb888():
        return QImage.Format_RGB888

try:
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    KEEP_ASPECT = Qt.AspectRatioMode.KeepAspectRatio
    SMOOTH_TRANSFORM = Qt.TransformationMode.SmoothTransformation
except AttributeError:
    ALIGN_CENTER = Qt.AlignCenter
    KEEP_ASPECT = Qt.KeepAspectRatio
    SMOOTH_TRANSFORM = Qt.SmoothTransformation


@dataclass(slots=True)
class MeasurementConfig:
    mode: str
    port: str | None
    baud: int
    duration_s: float
    sample_rate_hz: float
    age: int | None
    sex: str
    calibration_sbp: float | None
    calibration_dbp: float | None
    csv_input: Path | None = None
    camera_index: int | None = None
    camera_width: int | None = None
    camera_height: int | None = None
    camera_fps: float | None = None
    camera_gain_value: float | None = None


class MetricCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        self.title = QLabel(title)
        self.value = QLabel("-")
        self.detail = QLabel("대기 중")
        self.value.setStyleSheet("font-size:22px; font-weight:700;")
        self.detail.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)

    def update_card(self, value: str, detail: str) -> None:
        self.value.setText(value)
        self.detail.setText(detail)

    def reset(self) -> None:
        self.update_card("-", "대기 중")


class MeasurementWorker(QObject):
    progress_text = Signal(str)
    result_ready = Signal(object, str, object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, config: MeasurementConfig) -> None:
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        output_dir = new_session_dir(self.config.mode)
        capture_path: Path | None = None
        camera_summary: dict[str, Any] | None = None
        paths: dict[str, str] = {"output_dir": str(output_dir), "log_path": str(LOG_PATH)}
        try:
            self.progress_text.emit("측정을 시작합니다.")
            if self.config.mode == "csv":
                if self.config.csv_input is None:
                    raise ValueError("CSV 파일을 선택해주세요.")
                dataset = load_dataset_from_csv(self.config.csv_input, fallback_sample_rate_hz=self.config.sample_rate_hz)
                auto_camera_summary = self.config.csv_input.parent / "camera_rppg_summary.json"
                camera_summary = load_camera_summary(auto_camera_summary)
                if camera_summary is not None:
                    paths["camera_summary_json"] = str(auto_camera_summary)
            elif self.config.mode == "ppg":
                if not self.config.port:
                    raise ValueError("시리얼 포트를 선택해주세요.")
                samples = capture_serial_session(
                    port=self.config.port,
                    baud=self.config.baud,
                    duration_s=self.config.duration_s,
                    fallback_sample_rate_hz=self.config.sample_rate_hz,
                    status_callback=self.progress_text.emit,
                )
                capture_path = output_dir / "capture.csv"
                write_capture_csv(capture_path, samples)
                dataset = load_dataset_from_csv(capture_path, fallback_sample_rate_hz=self.config.sample_rate_hz)
                paths["capture_csv"] = str(capture_path)
            elif self.config.mode == "multimodal":
                if not self.config.port:
                    raise ValueError("시리얼 포트를 선택해주세요.")
                if self.config.camera_index is None:
                    raise ValueError("카메라를 선택해주세요.")
                bundle = capture_multimodal_session(
                    port=self.config.port,
                    baud=self.config.baud,
                    duration_s=self.config.duration_s,
                    fallback_sample_rate_hz=self.config.sample_rate_hz,
                    output_dir=output_dir,
                    camera_index=self.config.camera_index,
                    camera_width=self.config.camera_width,
                    camera_height=self.config.camera_height,
                    camera_fps=self.config.camera_fps,
                    camera_gain_value=self.config.camera_gain_value,
                    status_callback=self.progress_text.emit,
                )
                capture_path = Path(bundle["capture_csv_path"])
                dataset = load_dataset_from_csv(capture_path, fallback_sample_rate_hz=self.config.sample_rate_hz)
                camera_features = extract_camera_rppg_features(
                    Path(bundle["video_path"]),
                    output_dir,
                    frame_timestamps_path=Path(bundle["frame_csv_path"]),
                    status_callback=self.progress_text.emit,
                )
                camera_summary = camera_features["summary"]
                paths["capture_csv"] = str(bundle["capture_csv_path"])
                paths["camera_video"] = str(bundle["video_path"])
                paths["camera_features_csv"] = str(camera_features["features_csv_path"])
                paths["camera_summary_json"] = str(camera_features["summary_json_path"])
            else:
                raise ValueError(f"지원하지 않는 측정 모드입니다: {self.config.mode}")

            profile = build_user_profile(self.config.age, self.config.sex, self.config.calibration_sbp, self.config.calibration_dbp)
            report = run_analysis(dataset, profile, camera_summary=camera_summary)
            report_path, summary_path = write_report_files(output_dir, report, capture_path=capture_path, extra_paths=paths)
            paths["report_path"] = str(report_path)
            paths["summary_path"] = str(summary_path)
            self.result_ready.emit(report, format_summary_text(report), paths)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PSL_Test Cardiovascular Studio")
        self.resize(1360, 920)
        self._thread: QThread | None = None
        self._worker: MeasurementWorker | None = None
        self._camera_preview: cv2.VideoCapture | None = None
        self._port_entries: list[dict[str, str]] = []
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(33)
        self._preview_timer.timeout.connect(self.update_preview_frame)
        self.metric_cards: dict[str, MetricCard] = {}
        self._build_ui()
        self.refresh_ports()
        self.preview_label.setText("카메라 새로고침 버튼으로 장치를 검색하세요.")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        splitter = QSplitter()
        root_layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        form_group = QGroupBox("설정")
        form = QFormLayout(form_group)

        self.port_combo = QComboBox()
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.restart_preview)
        self.refresh_ports_btn = QPushButton("포트 새로고침")
        self.refresh_ports_btn.clicked.connect(self.refresh_ports)
        self.refresh_cameras_btn = QPushButton("카메라 새로고침")
        self.refresh_cameras_btn.clicked.connect(self.refresh_cameras)

        port_row = QWidget()
        port_row_l = QHBoxLayout(port_row)
        port_row_l.setContentsMargins(0, 0, 0, 0)
        port_row_l.addWidget(self.port_combo, 1)
        port_row_l.addWidget(self.refresh_ports_btn)

        camera_row = QWidget()
        camera_row_l = QHBoxLayout(camera_row)
        camera_row_l.setContentsMargins(0, 0, 0, 0)
        camera_row_l.addWidget(self.camera_combo, 1)
        camera_row_l.addWidget(self.refresh_cameras_btn)

        self.baud_spin = QSpinBox()
        self.baud_spin.setRange(9600, 2_000_000)
        self.baud_spin.setValue(1_000_000)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(5.0, 300.0)
        self.duration_spin.setValue(60.0)
        self.duration_spin.setSuffix(" s")
        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setRange(10.0, 1000.0)
        self.sample_rate_spin.setValue(250.0)
        self.sample_rate_spin.setSuffix(" Hz")
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 120)
        self.age_spin.setSpecialValueText("미입력")
        self.sex_combo = QComboBox()
        self.sex_combo.addItem("Unknown", "unknown")
        self.sex_combo.addItem("Male", "male")
        self.sex_combo.addItem("Female", "female")
        self.sbp_input = QLineEdit()
        self.dbp_input = QLineEdit()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 7680)
        self.width_spin.setValue(3840)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 4320)
        self.height_spin.setValue(2160)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.0, 240.0)
        self.fps_spin.setValue(30.0)
        self.gain_input = QLineEdit()

        self.csv_input = QLineEdit()
        self.csv_btn = QPushButton("CSV 선택")
        self.csv_btn.clicked.connect(self.browse_csv)
        csv_row = QWidget()
        csv_row_l = QHBoxLayout(csv_row)
        csv_row_l.setContentsMargins(0, 0, 0, 0)
        csv_row_l.addWidget(self.csv_input, 1)
        csv_row_l.addWidget(self.csv_btn)

        form.addRow("Serial Port", port_row)
        form.addRow("Camera", camera_row)
        form.addRow("Baud", self.baud_spin)
        form.addRow("Duration", self.duration_spin)
        form.addRow("Sample Rate", self.sample_rate_spin)
        form.addRow("Age", self.age_spin)
        form.addRow("Sex", self.sex_combo)
        form.addRow("Cal SBP", self.sbp_input)
        form.addRow("Cal DBP", self.dbp_input)
        form.addRow("Cam Width", self.width_spin)
        form.addRow("Cam Height", self.height_spin)
        form.addRow("Cam FPS", self.fps_spin)
        form.addRow("Cam Gain", self.gain_input)
        form.addRow("CSV", csv_row)
        left_layout.addWidget(form_group)

        action_group = QGroupBox("실행")
        action_layout = QGridLayout(action_group)
        self.ppg_btn = QPushButton("PPG 측정 시작")
        self.multi_btn = QPushButton("카메라+PPG 시작")
        self.csv_run_btn = QPushButton("CSV 분석")
        self.output_btn = QPushButton("결과 폴더 열기")
        self.ppg_btn.clicked.connect(lambda: self.start_measurement("ppg"))
        self.multi_btn.clicked.connect(lambda: self.start_measurement("multimodal"))
        self.csv_run_btn.clicked.connect(lambda: self.start_measurement("csv"))
        self.output_btn.clicked.connect(self.open_last_output_dir)
        action_layout.addWidget(self.ppg_btn, 0, 0)
        action_layout.addWidget(self.multi_btn, 0, 1)
        action_layout.addWidget(self.csv_run_btn, 1, 0)
        action_layout.addWidget(self.output_btn, 1, 1)
        left_layout.addWidget(action_group)

        preview_group = QGroupBox("카메라 미리보기")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("카메라 대기 중")
        self.preview_label.setAlignment(ALIGN_CENTER)
        self.preview_label.setMinimumSize(480, 320)
        preview_layout.addWidget(self.preview_label)
        left_layout.addWidget(preview_group, 1)

        self.status_label = QLabel("대기 중")
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right = QWidget()
        right_layout = QVBoxLayout(right)

        cards_group = QGroupBox("측정값")
        cards_layout = QGridLayout(cards_group)
        titles = {
            "heart_rate": "심박수",
            "hrv": "HRV",
            "stress": "스트레스",
            "circulation": "혈류/순환",
            "vascular_health": "혈관 건강",
            "vascular_age": "혈관 나이",
            "blood_pressure": "혈압 추정",
            "signal_quality": "신호 품질",
        }
        positions = [
            ("heart_rate", 0, 0), ("hrv", 0, 1), ("stress", 1, 0), ("circulation", 1, 1),
            ("vascular_health", 2, 0), ("vascular_age", 2, 1), ("blood_pressure", 3, 0), ("signal_quality", 3, 1),
        ]
        for key, row, col in positions:
            card = MetricCard(titles[key])
            self.metric_cards[key] = card
            cards_layout.addWidget(card, row, col)
        right_layout.addWidget(cards_group)

        summary_group = QGroupBox("요약")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_output = QPlainTextEdit()
        self.summary_output.setReadOnly(True)
        summary_layout.addWidget(self.summary_output)
        right_layout.addWidget(summary_group)

        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        right_layout.addWidget(log_group)

        path_group = QGroupBox("결과 파일")
        path_layout = QFormLayout(path_group)
        self.report_path = QLabel("-")
        self.summary_path = QLabel("-")
        self.capture_path = QLabel("-")
        self.camera_path = QLabel("-")
        self.log_path = QLabel(str(LOG_PATH))
        for widget in (self.report_path, self.summary_path, self.capture_path, self.camera_path, self.log_path):
            widget.setWordWrap(True)
        path_layout.addRow("Report", self.report_path)
        path_layout.addRow("Summary", self.summary_path)
        path_layout.addRow("Capture", self.capture_path)
        path_layout.addRow("Camera", self.camera_path)
        path_layout.addRow("Log", self.log_path)
        right_layout.addWidget(path_group)

        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes([520, 780])
        self.setCentralWidget(root)
        self.reset_cards()

    def reset_cards(self) -> None:
        for card in self.metric_cards.values():
            card.reset()

    def append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text)

    def refresh_ports(self) -> None:
        self.port_combo.clear()
        ports = list_serial_ports()
        self._port_entries = sorted(
            ports,
            key=lambda item: (
                0 if "arduino" in item["description"].lower() else 1,
                0 if "usb" in item["description"].lower() else 1,
                1 if "bluetooth" in item["description"].lower() else 0,
                item["device"],
            ),
        )
        if not self._port_entries:
            self.port_combo.addItem("포트 없음", None)
            return
        preferred_index = 0
        for index, port in enumerate(self._port_entries):
            self.port_combo.addItem(f"{port['device']} - {port['description']}", port)
            if "arduino" in port["description"].lower():
                preferred_index = index
        self.port_combo.setCurrentIndex(preferred_index)

    def refresh_cameras(self) -> None:
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        cameras = probe_camera_indices()
        if not cameras:
            self.camera_combo.addItem("카메라 없음", None)
        for camera in cameras:
            self.camera_combo.addItem(
                f"Camera {camera['index']} - {int(camera['width'])}x{int(camera['height'])} / {float(camera['fps']):.1f} fps",
                int(camera["index"]),
            )
        self.camera_combo.blockSignals(False)
        self.restart_preview()

    def browse_csv(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "CSV 파일 선택", str(Path.cwd()), "CSV Files (*.csv)")
        if file_path:
            self.csv_input.setText(file_path)

    def _optional_float(self, text: str) -> float | None:
        value = text.strip()
        if not value:
            return None
        return float(value)

    def _build_config(self, mode: str) -> MeasurementConfig:
        csv_text = self.csv_input.text().strip()
        csv_path = Path(csv_text).resolve() if csv_text else None
        age = None if self.age_spin.value() == 0 else int(self.age_spin.value())
        selected_port = self.port_combo.currentData()
        return MeasurementConfig(
            mode=mode,
            port=selected_port["device"] if isinstance(selected_port, dict) else None,
            baud=int(self.baud_spin.value()),
            duration_s=float(self.duration_spin.value()),
            sample_rate_hz=float(self.sample_rate_spin.value()),
            age=age,
            sex=str(self.sex_combo.currentData() or self.sex_combo.currentText()),
            calibration_sbp=self._optional_float(self.sbp_input.text()),
            calibration_dbp=self._optional_float(self.dbp_input.text()),
            csv_input=csv_path,
            camera_index=self.camera_combo.currentData(),
            camera_width=int(self.width_spin.value()) or None,
            camera_height=int(self.height_spin.value()) or None,
            camera_fps=float(self.fps_spin.value()) or None,
            camera_gain_value=self._optional_float(self.gain_input.text()),
        )

    def start_measurement(self, mode: str) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "작업 중", "이미 실행 중인 작업이 있습니다.")
            return
        try:
            config = self._build_config(mode)
            selected_port = self.port_combo.currentData()
            if mode in {"ppg", "multimodal"} and not config.port:
                raise ValueError("시리얼 포트를 선택해주세요.")
            if mode in {"ppg", "multimodal"} and isinstance(selected_port, dict):
                description = str(selected_port.get("description") or "").lower()
                if "bluetooth" in description:
                    arduino_ports = [item for item in self._port_entries if "arduino" in item["description"].lower()]
                    if arduino_ports:
                        raise ValueError(
                            f"현재 선택된 {selected_port['device']}는 Bluetooth serial link입니다. "
                            f"아두이노 포트 {arduino_ports[0]['device']}를 선택해주세요."
                        )
            if mode == "multimodal" and config.camera_index is None:
                raise ValueError("카메라를 선택해주세요.")
            if mode == "csv" and config.csv_input is None:
                raise ValueError("CSV 파일을 선택해주세요.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "설정 오류", str(exc))
            return

        self.stop_preview()
        self.reset_cards()
        self.summary_output.clear()
        self.log_output.clear()
        self.report_path.setText("-")
        self.summary_path.setText("-")
        self.capture_path.setText("-")
        self.camera_path.setText("-")
        self.status_label.setText("실행 중")
        self.ppg_btn.setEnabled(False)
        self.multi_btn.setEnabled(False)
        self.csv_run_btn.setEnabled(False)

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
        self._thread.finished.connect(self.cleanup_after_run)
        self._thread.start()

    @Slot()
    def cleanup_after_run(self) -> None:
        self.ppg_btn.setEnabled(True)
        self.multi_btn.setEnabled(True)
        self.csv_run_btn.setEnabled(True)
        self._thread = None
        self._worker = None
        self.start_preview()

    @Slot(object, str, object)
    def handle_result(self, report: dict[str, Any], summary_text: str, paths: dict[str, str]) -> None:
        self.status_label.setText("완료")
        self.summary_output.setPlainText(summary_text)
        self.report_path.setText(paths.get("report_path", "-"))
        self.summary_path.setText(paths.get("summary_path", "-"))
        self.capture_path.setText(paths.get("capture_csv", "-"))
        self.camera_path.setText(paths.get("camera_video", paths.get("camera_summary_json", "-")))
        self.update_cards(report)

    @Slot(str)
    def handle_failure(self, message: str) -> None:
        self.status_label.setText("실패")
        self.append_log(f"오류: {message}")
        QMessageBox.critical(self, "측정 실패", message)

    def update_cards(self, report: dict[str, Any]) -> None:
        metadata = report.get("metadata") or {}
        hr = report.get("heart_rate") or {}
        hrv = report.get("hrv") or {}
        stress = report.get("stress") or {}
        circulation = report.get("circulation") or {}
        vascular_health = report.get("vascular_health") or {}
        vascular_age = report.get("vascular_age") or {}
        bp = report.get("blood_pressure") or {}
        self.metric_cards["heart_rate"].update_card(f"{float(hr.get('heart_rate_bpm') or 0.0):.1f} bpm", f"peak {int(hr.get('peak_count') or 0)}")
        self.metric_cards["hrv"].update_card(f"{float(hrv.get('hrv_score') or 0.0):.1f}", f"RMSSD {float(hrv.get('rmssd_ms') or 0.0):.1f} | LF/HF {float(hrv.get('lf_hf_ratio') or 0.0):.2f}")
        self.metric_cards["stress"].update_card(f"{float(stress.get('stress_score') or 0.0):.1f}", str(stress.get("stress_state") or "-"))
        self.metric_cards["circulation"].update_card(f"{float(circulation.get('circulation_score') or 0.0):.1f}", f"PI {float(circulation.get('perfusion_index') or 0.0):.3f}")
        self.metric_cards["vascular_health"].update_card(f"{float(vascular_health.get('vascular_health_score') or 0.0):.1f}", f"RI {vascular_health.get('reflection_index')}")
        self.metric_cards["vascular_age"].update_card(f"{float(vascular_age.get('vascular_age_estimate') or 0.0):.1f} 세", f"gap {float(vascular_age.get('vascular_age_gap') or 0.0):+.1f}")
        self.metric_cards["blood_pressure"].update_card(f"{float(bp.get('estimated_sbp') or 0.0):.0f}/{float(bp.get('estimated_dbp') or 0.0):.0f}", str(bp.get("blood_pressure_trend") or "-"))
        self.metric_cards["signal_quality"].update_card(f"{float(metadata.get('signal_quality_score') or 0.0):.1f}", str(metadata.get("measurement_mode_label") or "-"))

    def start_preview(self) -> None:
        if self._thread is not None:
            return
        self.stop_preview()
        camera_index = self.camera_combo.currentData()
        if camera_index is None:
            self.preview_label.setText("카메라 없음")
            return
        self._camera_preview = open_camera_capture(
            int(camera_index),
            width=int(self.width_spin.value()) or None,
            height=int(self.height_spin.value()) or None,
            fps=float(self.fps_spin.value()) or None,
        )
        if self._camera_preview is None:
            self.preview_label.setText("카메라 열기 실패")
            return
        self._preview_timer.start()

    def stop_preview(self) -> None:
        self._preview_timer.stop()
        if self._camera_preview is not None:
            self._camera_preview.release()
            self._camera_preview = None

    def restart_preview(self) -> None:
        self.start_preview()

    @Slot()
    def update_preview_frame(self) -> None:
        if self._camera_preview is None:
            return
        ok, frame = self._camera_preview.read()
        if not ok or frame is None:
            self.preview_label.setText("카메라 프레임 없음")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], image_format_rgb888())
        pixmap = QPixmap.fromImage(image).scaled(self.preview_label.size(), KEEP_ASPECT, SMOOTH_TRANSFORM)
        self.preview_label.setPixmap(pixmap)

    def open_last_output_dir(self) -> None:
        target = self.report_path.text().strip()
        output_path = Path(target).resolve().parent if target and target != "-" else (Path(__file__).resolve().parent / "outputs").resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_path)
        if sys.platform.startswith("win"):
            os.startfile(output_dir)  # type: ignore[attr-defined]
        else:
            QMessageBox.information(self, "결과 폴더", output_dir)

    def closeEvent(self, event: Any) -> None:
        self.stop_preview()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
