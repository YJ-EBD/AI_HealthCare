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

from capture_and_analyze import (
    build_user_profile,
    capture_serial_session,
    load_dataset_from_csv,
    write_capture_csv,
    write_report_files,
)
from camera_rppg_features import extract_camera_rppg_features
from multimodal_capture import capture_multimodal_session, open_camera_capture, probe_camera_indices
from sequential_measurement_session import format_console_summary, run_stepwise_analysis

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


def list_serial_port_rows() -> list[object]:
    return list(list_ports.comports())


def describe_serial_port(port: object) -> str:
    description = getattr(port, "description", "") or "Serial Port"
    manufacturer = getattr(port, "manufacturer", "") or ""
    extras = f" / {manufacturer}" if manufacturer else ""
    return f"{port.device} - {description}{extras}"


def describe_camera_entry(camera_info: dict[str, int | float]) -> str:
    width = int(camera_info.get("width", 0))
    height = int(camera_info.get("height", 0))
    fps = float(camera_info.get("fps", 0.0))
    fps_text = f"{fps:.1f} fps" if fps > 0.0 else "fps unknown"
    return f"Camera {camera_info['index']} - {width}x{height} / {fps_text}"


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
        extra_paths: dict[str, str] = {}

        try:
            if self.config.csv_input is not None:
                self.progress_text.emit("Loading existing CSV input for analysis.")
                dataset = load_dataset_from_csv(self.config.csv_input, fallback_sample_rate_hz=self.config.sample_rate_hz)
            elif self.config.use_camera_dataset:
                if not self.config.port:
                    raise ValueError("A serial port is required for live multimodal capture.")
                if self.config.camera_index is None:
                    raise ValueError("A camera index must be selected for multimodal capture.")

                self.progress_text.emit("Starting multimodal capture: camera + iPPG.")
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
                    status_callback=self.progress_text.emit,
                )
                capture_path = Path(capture_bundle["capture_csv_path"])
                extra_paths = {
                    "camera_video": str(capture_bundle["video_path"]),
                    "camera_frames_csv": str(capture_bundle["frame_csv_path"]),
                    "session_manifest": str(capture_bundle["manifest_path"]),
                }
                self.progress_text.emit("Extracting camera rPPG features from the recorded video.")
                camera_features = extract_camera_rppg_features(
                    Path(capture_bundle["video_path"]),
                    output_dir,
                    frame_timestamps_path=Path(capture_bundle["frame_csv_path"]),
                    status_callback=self.progress_text.emit,
                )
                extra_paths["camera_features_csv"] = str(camera_features["features_csv_path"])
                extra_paths["camera_summary_json"] = str(camera_features["summary_json_path"])
                dataset = load_dataset_from_csv(capture_path, fallback_sample_rate_hz=self.config.sample_rate_hz)
                self.progress_text.emit("Multimodal capture complete. Starting 1.1 to 1.7 analysis.")
            else:
                if not self.config.port:
                    raise ValueError("A serial port is required unless CSV analysis is used.")
                self.progress_text.emit("Step 0/7: capturing raw PPG data.")
                samples = capture_serial_session(
                    self.config.port,
                    self.config.baud,
                    self.config.duration_s,
                    self.config.sample_rate_hz,
                    status_callback=self.progress_text.emit,
                )
                capture_path = output_dir / "capture.csv"
                write_capture_csv(capture_path, samples)
                dataset = load_dataset_from_csv(capture_path, fallback_sample_rate_hz=self.config.sample_rate_hz)
                self.progress_text.emit("Signal capture complete. Starting 1.1 to 1.7 analysis.")

            profile_args = SimpleNamespace(
                age=self.config.age,
                sex=self.config.sex,
                calibration_sbp=self.config.calibration_sbp,
                calibration_dbp=self.config.calibration_dbp,
            )
            report = run_stepwise_analysis(
                dataset,
                build_user_profile(profile_args),
                progress=lambda index, label: self.progress_text.emit(f"[{index}/7] {label} complete"),
            )
            report_path, summary_path = write_report_files(output_dir, report, capture_path=capture_path)
            summary_text = format_console_summary(report)
            self.result_ready.emit(report, summary_text, str(report_path), str(summary_path), extra_paths)
        except Exception as exc:  # noqa: BLE001
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
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cardiovascular Autonomic Studio")
        self.resize(1080, 1920)
        self.setMinimumSize(900, 1400)

        self._thread: QThread | None = None
        self._worker: MeasurementWorker | None = None
        self._tool_thread: QThread | None = None
        self._camera_worker: CameraExtractionWorker | None = None
        self._camera_preview: cv2.VideoCapture | None = None
        self._last_report_path = ""
        self._last_summary_path = ""
        self._last_extra_paths: dict[str, str] = {}

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
        self.use_camera_checkbox = QCheckBox("Create multimodal dataset with camera")
        self.camera_width_spin = QSpinBox()
        self.camera_height_spin = QSpinBox()
        self.camera_fps_spin = QDoubleSpinBox()
        self.camera_preview_label = QLabel("Camera preview is stopped.")
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Idle")
        self.summary_output = QPlainTextEdit()
        self.log_output = QPlainTextEdit()
        self.report_path_label = QLabel("-")
        self.summary_path_label = QLabel("-")
        self.camera_video_label = QLabel("-")
        self.manifest_path_label = QLabel("-")
        self.camera_features_label = QLabel("-")
        self.camera_summary_label = QLabel("-")
        self.video_input = QLineEdit()
        self.frame_csv_input = QLineEdit()
        self.extraction_output_dir_input = QLineEdit(str(OUTPUTS_DIR))
        self.camera_tool_output = QPlainTextEdit()
        self.camera_tool_features_label = QLabel("-")
        self.camera_tool_summary_label = QLabel("-")
        self.camera_tool_selected_signal_label = QLabel("-")
        self.camera_tool_selected_hr_label = QLabel("-")
        self.start_camera_extraction_button = QPushButton("Extract camera rPPG from video")
        self.strategy_view = QPlainTextEdit()
        self.blueprint_view = QPlainTextEdit()
        self.start_live_button = QPushButton("Start live measurement")
        self.analyze_csv_button = QPushButton("Analyze CSV")
        self.refresh_ports_button = QPushButton("Refresh serial ports")
        self.refresh_cameras_button = QPushButton("Refresh cameras")
        self.preview_camera_button = QPushButton("Start preview")
        self.stop_preview_button = QPushButton("Stop preview")
        self.open_output_button = QPushButton("Open output folder")
        self.preview_timer = QTimer(self)

        self._build_ui()
        self.refresh_ports()
        self.refresh_cameras()
        self._load_reference_docs()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel("Cardiovascular Autonomic Studio")
        header.setStyleSheet("font-size: 26px; font-weight: 700;")
        subtitle = QLabel(
            "Capture iPPG, optionally capture a synchronized camera dataset, analyze 1.1 to 1.7, and review the product plan."
        )
        subtitle.setStyleSheet("color: #5a6772;")

        tabs = QTabWidget()
        tabs.addTab(self._build_measurement_tab(), "Measurement")
        tabs.addTab(self._build_camera_tools_tab(), "Camera Tools")
        tabs.addTab(self._build_strategy_tab(), "Strategy")

        layout.addWidget(header)
        layout.addWidget(subtitle)
        layout.addWidget(tabs)
        self.setCentralWidget(central)

        refresh_action = QAction("Refresh Serial Ports", self)
        refresh_action.triggered.connect(self.refresh_ports)
        self.addAction(refresh_action)

        self.preview_timer.setInterval(50)
        self.preview_timer.timeout.connect(self.update_preview_frame)

    def _build_measurement_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        settings_group = QGroupBox("Session Settings")
        settings_layout = QGridLayout(settings_group)

        self.baud_spin.setRange(9600, 2_000_000)
        self.baud_spin.setValue(1_000_000)
        self.duration_spin.setRange(5.0, 600.0)
        self.duration_spin.setValue(60.0)
        self.duration_spin.setSuffix(" s")
        self.sample_rate_spin.setRange(20.0, 1000.0)
        self.sample_rate_spin.setValue(200.0)
        self.sample_rate_spin.setSuffix(" Hz")
        self.sex_combo.addItems(["unknown", "male", "female"])
        self.csv_input.setPlaceholderText("offline CSV path")
        self.age_input.setPlaceholderText("optional")
        self.calibration_sbp_input.setPlaceholderText("optional")
        self.calibration_dbp_input.setPlaceholderText("optional")

        csv_browse_button = QToolButton()
        csv_browse_button.setText("Browse")
        csv_browse_button.clicked.connect(self.browse_csv)

        settings_layout.addWidget(QLabel("Serial port"), 0, 0)
        settings_layout.addWidget(self.port_combo, 0, 1)
        settings_layout.addWidget(self.refresh_ports_button, 0, 2)
        settings_layout.addWidget(QLabel("Baud"), 1, 0)
        settings_layout.addWidget(self.baud_spin, 1, 1)
        settings_layout.addWidget(QLabel("Duration"), 1, 2)
        settings_layout.addWidget(self.duration_spin, 1, 3)
        settings_layout.addWidget(QLabel("Sample rate"), 2, 0)
        settings_layout.addWidget(self.sample_rate_spin, 2, 1)
        settings_layout.addWidget(QLabel("Age"), 2, 2)
        settings_layout.addWidget(self.age_input, 2, 3)
        settings_layout.addWidget(QLabel("Sex"), 3, 0)
        settings_layout.addWidget(self.sex_combo, 3, 1)
        settings_layout.addWidget(QLabel("Calibration SBP"), 3, 2)
        settings_layout.addWidget(self.calibration_sbp_input, 3, 3)
        settings_layout.addWidget(QLabel("Calibration DBP"), 4, 0)
        settings_layout.addWidget(self.calibration_dbp_input, 4, 1)
        settings_layout.addWidget(QLabel("CSV input"), 4, 2)
        csv_row = QHBoxLayout()
        csv_row.setContentsMargins(0, 0, 0, 0)
        csv_row.addWidget(self.csv_input)
        csv_row.addWidget(csv_browse_button)
        settings_layout.addLayout(csv_row, 4, 3)

        camera_group = QGroupBox("Camera Dataset")
        camera_layout = QGridLayout(camera_group)

        self.camera_width_spin.setRange(0, 7680)
        self.camera_width_spin.setValue(1920)
        self.camera_height_spin.setRange(0, 4320)
        self.camera_height_spin.setValue(1080)
        self.camera_fps_spin.setRange(0.0, 240.0)
        self.camera_fps_spin.setValue(30.0)
        self.camera_fps_spin.setSuffix(" fps")

        camera_layout.addWidget(self.use_camera_checkbox, 0, 0, 1, 2)
        camera_layout.addWidget(QLabel("Camera"), 1, 0)
        camera_layout.addWidget(self.camera_combo, 1, 1)
        camera_layout.addWidget(self.refresh_cameras_button, 1, 2)
        camera_layout.addWidget(QLabel("Width"), 2, 0)
        camera_layout.addWidget(self.camera_width_spin, 2, 1)
        camera_layout.addWidget(QLabel("Height"), 2, 2)
        camera_layout.addWidget(self.camera_height_spin, 2, 3)
        camera_layout.addWidget(QLabel("FPS"), 3, 0)
        camera_layout.addWidget(self.camera_fps_spin, 3, 1)
        camera_layout.addWidget(self.preview_camera_button, 3, 2)
        camera_layout.addWidget(self.stop_preview_button, 3, 3)

        preview_group = QGroupBox("Preview")
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

        status_group = QGroupBox("Run Status")
        status_layout = QFormLayout(status_group)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        status_layout.addRow("State", self.status_label)
        status_layout.addRow("Progress", self.progress_bar)
        status_layout.addRow("Report JSON", self.report_path_label)
        status_layout.addRow("Summary TXT", self.summary_path_label)
        status_layout.addRow("Camera video", self.camera_video_label)
        status_layout.addRow("Manifest", self.manifest_path_label)
        status_layout.addRow("Camera features", self.camera_features_label)
        status_layout.addRow("Camera summary", self.camera_summary_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlaceholderText("Analysis summary will appear here.")
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Run log will appear here.")
        self.summary_output.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self.log_output.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        splitter.addWidget(self._wrap_panel("Analysis Summary", self.summary_output))
        splitter.addWidget(self._wrap_panel("Run Log", self.log_output))
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

        tool_group = QGroupBox("Offline Camera rPPG Extraction")
        tool_layout = QGridLayout(tool_group)

        self.video_input.setPlaceholderText("Path to camera_rgb.mp4")
        self.frame_csv_input.setPlaceholderText("Optional path to camera_frames.csv")
        self.extraction_output_dir_input.setPlaceholderText("Output directory")

        browse_video_button = QToolButton()
        browse_video_button.setText("Video")
        browse_video_button.clicked.connect(self.browse_video)

        browse_frame_csv_button = QToolButton()
        browse_frame_csv_button.setText("Frames CSV")
        browse_frame_csv_button.clicked.connect(self.browse_frame_csv)

        browse_output_dir_button = QToolButton()
        browse_output_dir_button.setText("Output")
        browse_output_dir_button.clicked.connect(self.browse_extraction_output_dir)

        tool_layout.addWidget(QLabel("Video"), 0, 0)
        tool_layout.addWidget(self.video_input, 0, 1)
        tool_layout.addWidget(browse_video_button, 0, 2)
        tool_layout.addWidget(QLabel("Frame CSV"), 1, 0)
        tool_layout.addWidget(self.frame_csv_input, 1, 1)
        tool_layout.addWidget(browse_frame_csv_button, 1, 2)
        tool_layout.addWidget(QLabel("Output dir"), 2, 0)
        tool_layout.addWidget(self.extraction_output_dir_input, 2, 1)
        tool_layout.addWidget(browse_output_dir_button, 2, 2)
        tool_layout.addWidget(self.start_camera_extraction_button, 3, 0, 1, 3)

        result_group = QGroupBox("Camera Extraction Result")
        result_layout = QFormLayout(result_group)
        result_layout.addRow("Features CSV", self.camera_tool_features_label)
        result_layout.addRow("Summary JSON", self.camera_tool_summary_label)
        result_layout.addRow("Selected signal", self.camera_tool_selected_signal_label)
        result_layout.addRow("Estimated HR", self.camera_tool_selected_hr_label)

        self.camera_tool_output.setReadOnly(True)
        self.camera_tool_output.setPlaceholderText("Camera extraction log will appear here.")
        self.camera_tool_output.setWordWrapMode(QTextOption.WrapMode.NoWrap)

        layout.addWidget(tool_group)
        layout.addWidget(result_group)
        layout.addWidget(self._wrap_panel("Camera Extraction Log", self.camera_tool_output), 1)

        self.start_camera_extraction_button.clicked.connect(self.start_camera_extraction)
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
        strategy_splitter.addWidget(self._wrap_panel("Short Summary", self.strategy_view))
        strategy_splitter.addWidget(self._wrap_panel("Detailed Blueprint", self.blueprint_view))
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
            return f"Unable to read {path}:\n{exc}"

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
            self.port_combo.addItem("No serial port detected", None)

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
            self.camera_combo.addItem("No camera detected", None)

    def browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose CSV input",
            str(OUTPUTS_DIR),
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.csv_input.setText(path)

    def browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose camera video",
            str(OUTPUTS_DIR),
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
        )
        if path:
            self.video_input.setText(path)

    def browse_frame_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose camera frame CSV",
            str(OUTPUTS_DIR),
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.frame_csv_input.setText(path)

    def browse_extraction_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output directory", self.extraction_output_dir_input.text().strip() or str(OUTPUTS_DIR))
        if path:
            self.extraction_output_dir_input.setText(path)

    def start_preview(self) -> None:
        camera_index = self.camera_combo.currentData()
        if camera_index is None:
            QMessageBox.information(self, "No camera", "No camera is currently available.")
            return

        self.stop_preview()
        self._camera_preview = open_camera_capture(int(camera_index))
        if self._camera_preview is None:
            QMessageBox.critical(self, "Preview failed", f"Unable to open camera index {camera_index}.")
            return

        self._camera_preview.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.camera_width_spin.value()))
        self._camera_preview.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.camera_height_spin.value()))
        self._camera_preview.set(cv2.CAP_PROP_FPS, float(self.camera_fps_spin.value()))
        self.preview_timer.start()
        self.append_log(f"Camera preview started on index {camera_index}.")

    def stop_preview(self) -> None:
        self.preview_timer.stop()
        if self._camera_preview is not None:
            self._camera_preview.release()
            self._camera_preview = None
        self.camera_preview_label.setPixmap(QPixmap())
        self.camera_preview_label.setText("Camera preview is stopped.")

    @Slot()
    def update_preview_frame(self) -> None:
        if self._camera_preview is None:
            return

        ok, frame = self._camera_preview.read()
        if not ok or frame is None:
            return

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
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return
        self._start_worker(config)

    def start_csv_analysis(self) -> None:
        try:
            config = self._build_config(csv_mode=True)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return
        self._start_worker(config)

    def start_camera_extraction(self) -> None:
        if self._tool_thread is not None:
            QMessageBox.information(self, "Busy", "A camera extraction job is already running.")
            return

        video_text = self.video_input.text().strip()
        if not video_text:
            QMessageBox.critical(self, "Invalid settings", "Choose a video file before starting extraction.")
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
            raise ValueError("Choose a CSV file before starting CSV analysis.")

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
            sex=self.sex_combo.currentText(),
            calibration_sbp=float(calibration_sbp_text) if calibration_sbp_text else None,
            calibration_dbp=float(calibration_dbp_text) if calibration_dbp_text else None,
            csv_input=csv_path,
            use_camera_dataset=bool(self.use_camera_checkbox.isChecked()) and not csv_mode,
            camera_index=int(camera_index) if camera_index is not None else None,
            camera_width=int(self.camera_width_spin.value()) or None,
            camera_height=int(self.camera_height_spin.value()) or None,
            camera_fps=float(self.camera_fps_spin.value()) or None,
        )

    def _start_worker(self, config: SessionConfig) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "A measurement or analysis is already running.")
            return

        self.summary_output.clear()
        self.log_output.clear()
        self.status_label.setText("Running")
        self.progress_bar.setRange(0, 0)
        self.report_path_label.setText("-")
        self.summary_path_label.setText("-")
        self.camera_video_label.setText("-")
        self.manifest_path_label.setText("-")
        self.camera_features_label.setText("-")
        self.camera_summary_label.setText("-")
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
        self.progress_bar.setValue(1 if self.status_label.text() == "Complete" else 0)
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
        self.status_label.setText("Complete")
        self.summary_output.setPlainText(summary_text)
        self.report_path_label.setText(report_path)
        self.summary_path_label.setText(summary_path)
        self._last_report_path = report_path
        self._last_summary_path = summary_path
        self._last_extra_paths = extra_paths

        if extra_paths.get("camera_video"):
            self.camera_video_label.setText(extra_paths["camera_video"])
        if extra_paths.get("session_manifest"):
            self.manifest_path_label.setText(extra_paths["session_manifest"])
        if extra_paths.get("camera_features_csv"):
            self.camera_features_label.setText(extra_paths["camera_features_csv"])
        if extra_paths.get("camera_summary_json"):
            self.camera_summary_label.setText(extra_paths["camera_summary_json"])

        warnings = report.get("warnings") or []
        if warnings:
            self.log_output.appendPlainText("")
            self.log_output.appendPlainText("Warnings:")
            for warning in warnings:
                self.log_output.appendPlainText(f"- {warning}")

    @Slot(str)
    def handle_failure(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.log_output.appendPlainText("")
        self.log_output.appendPlainText(f"ERROR: {message}")
        QMessageBox.critical(self, "Measurement failed", message)

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
        self.camera_tool_selected_signal_label.setText(selected_signal)
        self.camera_tool_selected_hr_label.setText(f"{selected_hr_bpm:.2f} bpm")
        self.camera_tool_output.appendPlainText("")
        self.camera_tool_output.appendPlainText("Camera extraction complete.")

    @Slot(str)
    def handle_camera_extraction_failure(self, message: str) -> None:
        self.camera_tool_output.appendPlainText("")
        self.camera_tool_output.appendPlainText(f"ERROR: {message}")
        QMessageBox.critical(self, "Camera extraction failed", message)

    def open_last_output_dir(self) -> None:
        target_path = ""
        if self._last_report_path:
            target_path = str(Path(self._last_report_path).resolve().parent)
        elif OUTPUTS_DIR.exists():
            target_path = str(OUTPUTS_DIR.resolve())

        if not target_path:
            QMessageBox.information(self, "No output", "There is no output folder to open yet.")
            return

        if sys.platform.startswith("win"):
            os.startfile(target_path)  # type: ignore[attr-defined]
        else:
            QMessageBox.information(self, "Output folder", target_path)

    def closeEvent(self, event: Any) -> None:
        self.stop_preview()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
