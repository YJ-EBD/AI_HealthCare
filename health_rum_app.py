from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


ROOT_DIR = Path(__file__).resolve().parent
PSL_TEST_DIR = ROOT_DIR / "PSL_Test"
FACE_AI_DIR = ROOT_DIR / "Face_AI"
HEALTH_RUM_OUTPUTS = ROOT_DIR / "Health_rum_outputs"

for import_path in (PSL_TEST_DIR, FACE_AI_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from analysis_pipeline import build_user_profile, run_analysis  # noqa: E402
from camera_rppg import capture_multimodal_session, open_camera_capture, probe_camera_indices  # noqa: E402
from live_runtime import LiveSkinAnalyzer, build_default_paths, draw_analysis_overlay, save_image_unicode  # noqa: E402
from serial_capture import capture_serial_session, list_serial_ports, load_dataset_from_csv, write_capture_csv  # noqa: E402
from health_rum_profile import (
    PROFILE_LIBRARY,
    SURVEY_GROUPS,
    build_profile_recommendation,
    format_profile_recommendation,
    format_survey_summary,
)


try:
    from PySide6.QtCore import QEasingCurve, QObject, QPoint, QPropertyAnimation, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen, QPixmap, QRadialGradient
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGraphicsOpacityEffect,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )

    def image_format_rgb888():
        return QImage.Format.Format_RGB888

except ImportError:
    from PyQt5.QtCore import QEasingCurve, QObject, QPoint, QPropertyAnimation, QThread, QTimer, Qt, pyqtSignal as Signal, pyqtSlot as Slot
    from PyQt5.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen, QPixmap, QRadialGradient
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGraphicsOpacityEffect,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )

    def image_format_rgb888():
        return QImage.Format_RGB888


try:
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    ALIGN_TOP = Qt.AlignmentFlag.AlignTop
    ALIGN_LEFT = Qt.AlignmentFlag.AlignLeft
    ALIGN_VCENTER = Qt.AlignmentFlag.AlignVCenter
    KEEP_ASPECT = Qt.AspectRatioMode.KeepAspectRatio
    SMOOTH_TRANSFORM = Qt.TransformationMode.SmoothTransformation
    SCROLLBAR_OFF = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
except AttributeError:
    ALIGN_CENTER = Qt.AlignCenter
    ALIGN_TOP = Qt.AlignTop
    ALIGN_LEFT = Qt.AlignLeft
    ALIGN_VCENTER = Qt.AlignVCenter
    KEEP_ASPECT = Qt.KeepAspectRatio
    SMOOTH_TRANSFORM = Qt.SmoothTransformation
    SCROLLBAR_OFF = Qt.ScrollBarAlwaysOff


STEP_LABELS = (
    "1. 시작",
    "2. 체질 설문",
    "3. PSL_Test",
    "4. 얼굴 촬영",
    "5. 얼굴 분석 확인",
    "6. 최종 결과",
)


def ensure_health_rum_dirs() -> None:
    HEALTH_RUM_OUTPUTS.mkdir(parents=True, exist_ok=True)


def new_health_rum_session_dir() -> Path:
    ensure_health_rum_dirs()
    session_dir = HEALTH_RUM_OUTPUTS / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)


def parse_optional_float(text: str) -> float | None:
    value = text.strip()
    if not value:
        return None
    return float(value)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def configure_opencv_logging() -> None:
    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:  # noqa: BLE001
        pass


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_psl_report(report: dict[str, Any] | None) -> str:
    if not report:
        return "아직 PSL_Test 결과가 없습니다."

    meta = report.get("metadata") or {}
    heart = report.get("heart_rate") or {}
    hrv = report.get("hrv") or {}
    stress = report.get("stress") or {}
    circulation = report.get("circulation") or {}
    vascular_health = report.get("vascular_health") or {}
    vascular_age = report.get("vascular_age") or {}
    blood_pressure = report.get("blood_pressure") or {}
    warnings = report.get("warnings") or []

    lines = [
        "PSL_Test 요약",
        "==============",
        f"측정 모드: {meta.get('measurement_mode_label') or meta.get('measurement_mode') or '알 수 없음'}",
        f"신호 품질: {safe_float(meta.get('signal_quality_score')):.1f} / 100",
        f"심박수: {safe_float(heart.get('heart_rate_bpm')):.1f} bpm",
        f"HRV: RMSSD {safe_float(hrv.get('rmssd_ms')):.1f} ms | SDNN {safe_float(hrv.get('sdnn_ms')):.1f} ms | pNN50 {safe_float(hrv.get('pnn50')):.1f}%",
        f"스트레스: {safe_float(stress.get('stress_score')):.1f} / {stress.get('stress_state') or '-'}",
        f"혈류 순환: {safe_float(circulation.get('circulation_score')):.1f} / 100 | 관류지수 {safe_float(circulation.get('perfusion_index')):.3f}",
        f"혈관 건강: {safe_float(vascular_health.get('vascular_health_score')):.1f} / 100",
        f"혈관 나이: {safe_float(vascular_age.get('vascular_age_estimate')):.1f} | 차이 {safe_float(vascular_age.get('vascular_age_gap')):+.1f}",
        f"혈압 추정: {safe_float(blood_pressure.get('estimated_sbp')):.0f}/{safe_float(blood_pressure.get('estimated_dbp')):.0f} mmHg",
    ]

    if warnings:
        lines.append("")
        lines.append("경고")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def summarize_face_result(face_result: dict[str, Any] | None) -> str:
    if not face_result:
        return "아직 Face_AI 결과가 없습니다."

    if not face_result.get("face_detected", True):
        return (
            "Face_AI가 얼굴을 감지하지 못했습니다.\n"
            f"메시지: {face_result.get('message') or '메시지 없음'}\n"
            "안내: 얼굴을 중앙에 맞추고 조금 더 가까이 촬영한 뒤 다시 시도하세요."
        )

    overall = face_result.get("overall_score")
    overall_label = face_result.get("overall_label") or "-"
    tasks = list((face_result.get("tasks") or {}).values())
    tasks.sort(key=lambda item: safe_float(item.get("normalized_score")), reverse=True)

    lines = [
        "Face_AI 요약",
        "============",
        f"얼굴 감지: {'예' if face_result.get('face_detected') else '아니오'}",
        f"종합 점수: {safe_float(overall):.1f} / {overall_label}" if overall is not None else "종합 점수: -",
        f"분석 모드: {face_result.get('calibration_mode') or '-'}",
        "",
        "주요 항목",
    ]
    if tasks:
        for item in tasks[:5]:
            lines.append(
                f"- {item.get('title') or '-'}: {safe_float(item.get('normalized_score')):.1f} / {item.get('severity_label') or '-'}"
            )
    else:
        lines.append("- 표시할 항목 점수가 없습니다.")
    return "\n".join(lines)


def top_face_concern(face_result: dict[str, Any] | None) -> tuple[str, str]:
    if not face_result:
        return "-", "Face_AI 결과 없음"
    tasks = list((face_result.get("tasks") or {}).values())
    tasks.sort(key=lambda item: safe_float(item.get("normalized_score")), reverse=True)
    if not tasks:
        return "-", "항목 데이터 없음"
    top_task = tasks[0]
    return f"{safe_float(top_task.get('normalized_score')):.1f}", str(top_task.get("title") or "주요 항목")


def detect_relaxed_face_box(frame_bgr) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return None

    search_configs = (
        (1.10, 4, (120, 120)),
        (1.08, 4, (96, 96)),
        (1.06, 3, (96, 96)),
    )
    for scale_factor, min_neighbors, min_size in search_configs:
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        if len(detections) > 0:
            x, y, w, h = max(detections, key=lambda item: item[2] * item[3])
            return int(x), int(y), int(w), int(h)
    return None


def analyze_face_with_fallback(analyzer: LiveSkinAnalyzer, frame_bgr) -> dict[str, Any]:
    result = analyzer.analyze_frame(frame_bgr)
    if result.get("face_detected"):
        result["fallback_face_detector_used"] = False
        return result

    fallback_box = detect_relaxed_face_box(frame_bgr)
    if fallback_box is not None:
        analyzer.last_face_box = fallback_box
        retry_result = analyzer.analyze_frame(frame_bgr)
        if retry_result.get("face_detected"):
            retry_result["fallback_face_detector_used"] = True
            retry_result["fallback_face_box"] = list(fallback_box)
            return retry_result

    result["fallback_face_detector_used"] = False
    if fallback_box is not None:
        result["fallback_face_box"] = list(fallback_box)
    return result


@dataclass(slots=True)
class HealthRumPslConfig:
    mode: str
    port: str | None
    duration_s: float
    age: int | None
    sex: str
    calibration_sbp: float | None
    calibration_dbp: float | None
    camera_index: int | None
    baud: int = 1_000_000
    sample_rate_hz: float = 250.0
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: float = 30.0


class FuturisticBackground(QWidget):
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor("#04121c"))
        gradient.setColorAt(0.45, QColor("#0b2233"))
        gradient.setColorAt(1.0, QColor("#12273d"))
        painter.fillRect(self.rect(), gradient)

        for center_x, center_y, radius, color_hex in (
            (self.width() * 0.15, self.height() * 0.12, 260, "#18c3ff"),
            (self.width() * 0.82, self.height() * 0.18, 210, "#f7a23b"),
            (self.width() * 0.50, self.height() * 0.78, 320, "#49e1b6"),
        ):
            radial = QRadialGradient(center_x, center_y, radius)
            glow = QColor(color_hex)
            glow.setAlpha(58)
            edge = QColor(color_hex)
            edge.setAlpha(0)
            radial.setColorAt(0.0, glow)
            radial.setColorAt(1.0, edge)
            painter.setBrush(radial)
            painter.setPen(Qt.NoPen if hasattr(Qt, "NoPen") else 0)
            painter.drawEllipse(QPoint(int(center_x), int(center_y)), int(radius), int(radius))

        grid_pen = QPen(QColor(177, 232, 255, 18))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        step = 42
        for x in range(0, self.width(), step):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            painter.drawLine(0, y, self.width(), y)

        super().paintEvent(event)


class AnimatedStackedWidget(QStackedWidget):
    def __init__(self) -> None:
        super().__init__()
        self._page_animation: QPropertyAnimation | None = None
        self._fade_animation: QPropertyAnimation | None = None

    def setCurrentIndexAnimated(self, index: int) -> None:  # noqa: N802
        if index == self.currentIndex():
            return

        next_widget = self.widget(index)
        if next_widget is None:
            return

        super().setCurrentIndex(index)
        base_pos = next_widget.pos()
        next_widget.move(base_pos + QPoint(24, 18))

        opacity = QGraphicsOpacityEffect(next_widget)
        next_widget.setGraphicsEffect(opacity)
        opacity.setOpacity(0.0)

        page_animation = QPropertyAnimation(next_widget, b"pos", self)
        page_animation.setDuration(320)
        page_animation.setStartValue(base_pos + QPoint(24, 18))
        page_animation.setEndValue(base_pos)
        page_animation.setEasingCurve(QEasingCurve.OutCubic)

        fade_animation = QPropertyAnimation(opacity, b"opacity", self)
        fade_animation.setDuration(320)
        fade_animation.setStartValue(0.0)
        fade_animation.setEndValue(1.0)
        fade_animation.setEasingCurve(QEasingCurve.OutCubic)

        def cleanup() -> None:
            next_widget.move(base_pos)
            next_widget.setGraphicsEffect(None)
            self._page_animation = None
            self._fade_animation = None

        fade_animation.finished.connect(cleanup)
        self._page_animation = page_animation
        self._fade_animation = fade_animation
        page_animation.start()
        fade_animation.start()


class StepBadge(QFrame):
    def __init__(self, step_no: int, title: str) -> None:
        super().__init__()
        self.setObjectName("StepBadge")
        self.step_no = QLabel(str(step_no))
        self.step_no.setObjectName("StepNumber")
        self.title_label = QLabel(title)
        self.title_label.setObjectName("StepTitle")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(self.step_no)
        layout.addWidget(self.title_label, 1)

    def set_state(self, state: str) -> None:
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)
        for child in (self.step_no, self.title_label):
            child.style().unpolish(child)
            child.style().polish(child)


class DashboardCard(QFrame):
    def __init__(self, title: str, accent: str) -> None:
        super().__init__()
        self.setObjectName("DashboardCard")
        self.setProperty("accent", accent)

        accent_bar = QFrame()
        accent_bar.setObjectName("AccentBar")
        accent_bar.setStyleSheet(f"background:{accent}; border-radius: 4px;")
        accent_bar.setFixedHeight(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("CardTitle")
        self.title_label.setStyleSheet(f"color:{accent};")
        self.title_label.setWordWrap(True)

        self.value_label = QLabel("-")
        self.value_label.setObjectName("CardValue")
        self.value_label.setWordWrap(True)

        self.detail_label = QLabel("측정 대기 중")
        self.detail_label.setObjectName("CardDetail")
        self.detail_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        layout.addWidget(accent_bar)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)

        self.setMinimumHeight(136)

    def update_card(self, value: str, detail: str) -> None:
        self.value_label.setText(value)
        self.detail_label.setText(detail)

    def reset(self) -> None:
        self.update_card("-", "측정 대기 중")


class ReportTableWidget(QFrame):
    def __init__(self, accent: str = "#18c3ff") -> None:
        super().__init__()
        self.setObjectName("ReportTable")
        self.accent = accent
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(10)

        self.caption_label = QLabel("")
        self.caption_label.setObjectName("ReportCaption")
        self.caption_label.setWordWrap(True)
        self.caption_label.hide()
        self._layout.addWidget(self.caption_label)

        self.table_frame = QFrame()
        self.table_frame.setObjectName("ReportTableFrame")
        self.grid = QGridLayout(self.table_frame)
        self.grid.setContentsMargins(1, 1, 1, 1)
        self.grid.setHorizontalSpacing(1)
        self.grid.setVerticalSpacing(1)
        self._layout.addWidget(self.table_frame)

    def _clear_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_cell(
        self,
        text: str,
        *,
        header: bool = False,
        emphasized: bool = False,
        alternate: bool = False,
    ) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(ALIGN_LEFT | ALIGN_TOP)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        if header:
            label.setObjectName("ReportHeaderCell")
            label.setStyleSheet(
                "padding: 12px 12px;"
                "font-size: 12px;"
                "font-weight: 700;"
                "color: #f6fbff;"
                "background: rgba(17, 57, 83, 0.96);"
                "border: 1px solid rgba(143, 214, 243, 0.20);"
            )
            return label

        base_bg = "rgba(9, 25, 39, 0.94)" if not alternate else "rgba(12, 31, 47, 0.96)"
        border = "rgba(143, 214, 243, 0.14)"
        if emphasized:
            base_bg = "rgba(24, 84, 73, 0.54)"
            border = "rgba(132, 246, 203, 0.30)"

        label.setObjectName("ReportBodyCell")
        label.setStyleSheet(
            "padding: 11px 12px;"
            "font-size: 12px;"
            "line-height: 1.45;"
            f"color: {'#f8fdff' if emphasized else '#d9edf7'};"
            f"background: {base_bg};"
            f"border: 1px solid {border};"
        )
        return label

    def set_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        caption: str | None = None,
        column_stretches: list[int] | None = None,
        emphasized_rows: set[int] | None = None,
    ) -> None:
        self._clear_grid()
        emphasized_rows = emphasized_rows or set()

        if caption:
            self.caption_label.setText(caption)
            self.caption_label.show()
        else:
            self.caption_label.hide()

        if not rows:
            rows = [["데이터 없음"] + [""] * max(0, len(headers) - 1)]

        for col, header_text in enumerate(headers):
            self.grid.addWidget(self._build_cell(header_text, header=True), 0, col)

        for row_idx, row in enumerate(rows, start=1):
            normalized = [str(item) for item in row]
            if len(normalized) < len(headers):
                normalized.extend([""] * (len(headers) - len(normalized)))
            for col, value in enumerate(normalized[: len(headers)]):
                self.grid.addWidget(
                    self._build_cell(
                        value,
                        header=False,
                        emphasized=(row_idx - 1) in emphasized_rows,
                        alternate=bool((row_idx - 1) % 2),
                    ),
                    row_idx,
                    col,
                )

        for col in range(len(headers)):
            stretch = 1
            if column_stretches and col < len(column_stretches):
                stretch = max(1, int(column_stretches[col]))
            self.grid.setColumnStretch(col, stretch)


class ReportSection(QFrame):
    def __init__(self, title: str, accent: str, subtitle: str | None = None) -> None:
        super().__init__()
        self.setObjectName("ReportSection")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        accent_bar = QFrame()
        accent_bar.setObjectName("ReportAccentBar")
        accent_bar.setFixedHeight(5)
        accent_bar.setStyleSheet(f"background:{accent}; border-radius: 2px;")
        layout.addWidget(accent_bar)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("ReportSectionTitle")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        self.subtitle_label = QLabel(subtitle or "")
        self.subtitle_label.setObjectName("ReportSectionSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        layout.addWidget(self.subtitle_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 4, 0, 0)
        self.content_layout.setSpacing(12)
        layout.addLayout(self.content_layout)

    def add_content(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)


class AspectRatioLabel(QLabel):
    def __init__(self, placeholder_text: str) -> None:
        super().__init__(placeholder_text)
        self.setObjectName("PreviewFrame")
        self.setAlignment(ALIGN_CENTER)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap: QPixmap | None = None
        self._placeholder_text = placeholder_text
        self._display_mode = "text"

    def set_placeholder(self, text: str | None = None) -> None:
        self._display_mode = "text"
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text or self._placeholder_text)

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        self._display_mode = "pixmap"
        self._pixmap = pixmap
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._display_mode != "pixmap" or self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), KEEP_ASPECT, SMOOTH_TRANSFORM)
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()


class HealthRumPslWorker(QObject):
    progress = Signal(str)
    completed = Signal(object, object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, config: HealthRumPslConfig, session_dir: Path) -> None:
        super().__init__()
        self.config = config
        self.session_dir = session_dir

    @Slot()
    def run(self) -> None:
        output_dir = self.session_dir / "psl_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        capture_path: Path | None = None
        paths: dict[str, str] = {"session_dir": str(self.session_dir), "output_dir": str(output_dir)}

        try:
            self.progress.emit("PSL_Test 측정을 시작합니다.")
            if self.config.mode == "multimodal":
                if not self.config.port:
                    raise ValueError("먼저 Arduino 시리얼 포트를 선택하세요.")
                if self.config.camera_index is None:
                    raise ValueError("카메라 + PPG 모드용 카메라를 선택하세요.")

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
                    status_callback=self.progress.emit,
                )
                capture_path = Path(bundle["capture_csv_path"])
                paths["capture_csv"] = str(bundle["capture_csv_path"])
                paths["camera_video"] = str(bundle["video_path"])
                paths["camera_frames_csv"] = str(bundle["frame_csv_path"])
                paths["manifest_path"] = str(bundle["manifest_path"])
            else:
                if not self.config.port:
                    raise ValueError("먼저 Arduino 시리얼 포트를 선택하세요.")

                samples = capture_serial_session(
                    port=self.config.port,
                    baud=self.config.baud,
                    duration_s=self.config.duration_s,
                    fallback_sample_rate_hz=self.config.sample_rate_hz,
                    status_callback=self.progress.emit,
                )
                capture_path = output_dir / "capture.csv"
                write_capture_csv(capture_path, samples)
                paths["capture_csv"] = str(capture_path)

            dataset = load_dataset_from_csv(Path(paths["capture_csv"]), fallback_sample_rate_hz=self.config.sample_rate_hz)
            profile = build_user_profile(
                self.config.age,
                self.config.sex,
                self.config.calibration_sbp,
                self.config.calibration_dbp,
            )
            report = run_analysis(dataset, profile)

            report_path = output_dir / "analysis_report.json"
            summary_path = output_dir / "summary.txt"
            payload = dict(report)
            if capture_path is not None:
                payload["capture_csv"] = str(capture_path)
            payload["extra_paths"] = paths

            write_json(report_path, payload)
            summary_path.write_text(summarize_psl_report(report), encoding="utf-8")

            paths["report_path"] = str(report_path)
            paths["summary_path"] = str(summary_path)
            self.completed.emit(report, paths)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class FaceAiAnalyzeWorker(QObject):
    progress = Signal(str)
    completed = Signal(object, object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, frame_bgr, session_dir: Path) -> None:
        super().__init__()
        self.frame_bgr = frame_bgr
        self.session_dir = session_dir

    @Slot()
    def run(self) -> None:
        output_dir = self.session_dir / "face_ai"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            defaults = build_default_paths()
            self.progress.emit("Face_AI 모델을 불러오는 중입니다.")
            analyzer = LiveSkinAnalyzer(checkpoint_root=defaults["checkpoint_root"])
            self.progress.emit("현재 얼굴 프레임을 분석하는 중입니다.")
            result = analyze_face_with_fallback(analyzer, self.frame_bgr)
            if result.get("fallback_face_detector_used"):
                self.progress.emit("기본 감지가 어려워 보조 얼굴 감지로 다시 분석했습니다.")
            annotated = draw_analysis_overlay(self.frame_bgr, result)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = output_dir / f"face_ai_snapshot_{timestamp}.jpg"
            result_path = output_dir / "face_ai_result.json"
            summary_path = output_dir / "face_ai_summary.txt"

            save_image_unicode(snapshot_path, annotated)
            write_json(result_path, result)
            summary_path.write_text(summarize_face_result(result), encoding="utf-8")

            paths = {
                "output_dir": str(output_dir),
                "snapshot_path": str(snapshot_path),
                "result_json": str(result_path),
                "summary_path": str(summary_path),
            }
            self.completed.emit(result, paths)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class HealthRumWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("헬스럼 스튜디오")
        self.resize(900, 1440)
        self.setMinimumSize(900, 1440)

        self.session_dir: Path | None = None
        self.survey_answers: dict[str, int] = {}
        self.survey_details: dict[str, list[str]] = {}
        self.survey_result: dict[str, Any] | None = None
        self.survey_completed = False
        self.psl_report: dict[str, Any] | None = None
        self.psl_paths: dict[str, str] = {}
        self.face_result: dict[str, Any] | None = None
        self.face_paths: dict[str, str] = {}

        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._face_capture: cv2.VideoCapture | None = None
        self._face_timer = QTimer(self)
        self._face_timer.setInterval(33)
        self._face_timer.timeout.connect(self.update_face_preview)
        self._current_face_frame = None
        self._port_entries: list[dict[str, str]] = []
        self._camera_entries: list[dict[str, int | float]] = []

        self.step_badges: list[StepBadge] = []
        self.survey_checkboxes: dict[str, list[QCheckBox]] = {}
        self.survey_count_labels: dict[str, QLabel] = {}
        self.psl_cards: dict[str, DashboardCard] = {}
        self.face_review_cards: dict[str, DashboardCard] = {}
        self.final_cards: dict[str, DashboardCard] = {}
        self.final_report_sections: list[QWidget] = []
        self._final_report_animations: list[tuple[QGraphicsOpacityEffect, QPropertyAnimation]] = []

        configure_opencv_logging()
        self.build_window()
        self.build_ui()
        self.apply_styles()
        self.refresh_ports()
        self.refresh_all_cameras()
        self.update_survey_preview()
        self.update_step_state(0)

    def build_window(self) -> None:
        self.setFont(QFont("Malgun Gothic", 10))

    def build_ui(self) -> None:
        root = FuturisticBackground()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        hero = QFrame()
        hero.setObjectName("Hero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(10)

        title = QLabel("헬스럼")
        title.setObjectName("HeroTitle")
        subtitle = QLabel(
            "바이오신호 측정, Face_AI 분석 확인, 최종 통합 리포트를 하나의 흐름으로 실행합니다. "
            "현재 마법사는 6단계로 구성됩니다: 시작 -> 체질 설문 -> PSL_Test -> 얼굴 촬영 -> 얼굴 분석 확인 -> 최종 결과"
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)

        steps_row = QWidget()
        steps_layout = QHBoxLayout(steps_row)
        steps_layout.setContentsMargins(0, 6, 0, 0)
        steps_layout.setSpacing(10)
        for index, label in enumerate(STEP_LABELS, start=1):
            badge = StepBadge(index, label)
            self.step_badges.append(badge)
            steps_layout.addWidget(badge, 1)
        hero_layout.addWidget(steps_row)
        root_layout.addWidget(hero)

        self.stacked = AnimatedStackedWidget()
        root_layout.addWidget(self.stacked, 1)

        self.page_main = self.wrap_page(self.build_main_page())
        self.page_survey = self.wrap_page(self.build_survey_page())
        self.page_psl = self.wrap_page(self.build_psl_page())
        self.page_face_capture = self.wrap_page(self.build_face_capture_page())
        self.page_face_review = self.wrap_page(self.build_face_review_page())
        self.page_final = self.wrap_page(self.build_final_page())

        for page in (
            self.page_main,
            self.page_survey,
            self.page_psl,
            self.page_face_capture,
            self.page_face_review,
            self.page_final,
        ):
            self.stacked.addWidget(page)

        self.setCentralWidget(root)

    def wrap_page(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(SCROLLBAR_OFF)
        scroll.setWidget(content)
        return scroll

    def build_main_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        intro_panel = QFrame()
        intro_panel.setObjectName("Panel")
        intro_layout = QVBoxLayout(intro_panel)
        intro_layout.setContentsMargins(22, 22, 22, 22)
        intro_layout.setSpacing(14)

        heading = QLabel("1. 세션 시작")
        heading.setObjectName("SectionTitle")
        body = QLabel(
            "새 세션을 만들고 바로 PSL_Test 측정으로 이동합니다. "
            "최종 결과 페이지에서는 PSL 바이오신호 결과와 Face_AI 결과가 한 화면에 통합되어 표시됩니다."
        )
        body.setObjectName("BodyText")
        body.setWordWrap(True)

        guide = QLabel(
            "권장 순서\n"
            "1. Arduino UNO와 PSL_iPPG2C를 연결합니다\n"
            "2. PSL_Test와 Face_AI에 사용할 카메라를 연결합니다\n"
            "3. 세션을 시작하고 6단계 흐름대로 진행합니다"
        )
        guide.setObjectName("InfoBlock")
        guide.setWordWrap(True)

        self.main_session_label = QLabel("아직 생성된 세션이 없습니다.")
        self.main_session_label.setObjectName("InfoText")

        self.main_start_button = QPushButton("세션 시작\n체질 설문으로 이동")
        self.main_start_button.clicked.connect(self.start_new_session)

        intro_layout.addWidget(heading)
        intro_layout.addWidget(body)
        intro_layout.addWidget(guide)
        intro_layout.addWidget(self.main_session_label)
        intro_layout.addWidget(self.main_start_button)
        layout.addWidget(intro_panel)

        feature_grid = QGridLayout()
        feature_grid.setSpacing(12)
        for idx, meta in enumerate(
            (
                ("고정 시리얼 경로", "PSL_Test용 Arduino COM 포트와 1000000 baud를 고정 사용합니다", "#18c3ff"),
                ("Face_AI 확인 단계", "최종 결과 전에 Face_AI 전용 확인 페이지가 추가되었습니다", "#49e1b6"),
                ("통합 저장", "모든 결과가 하나의 세션 요약으로 다시 저장됩니다", "#f7a23b"),
            )
        ):
            card = DashboardCard(meta[0], meta[2])
            card.update_card("준비됨", meta[1])
            feature_grid.addWidget(card, idx // 2, idx % 2)
        feature_group = QFrame()
        feature_group.setObjectName("Panel")
        fg_layout = QVBoxLayout(feature_group)
        fg_layout.setContentsMargins(16, 16, 16, 16)
        fg_layout.addLayout(feature_grid)
        layout.addWidget(feature_group)
        layout.addStretch(1)
        return page

    def build_survey_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        survey_panel = QFrame()
        survey_panel.setObjectName("Panel")
        survey_layout = QVBoxLayout(survey_panel)
        survey_layout.setContentsMargins(18, 18, 18, 18)
        survey_layout.setSpacing(12)

        title = QLabel("2. 4성 체질 설문")
        title.setObjectName("SectionTitle")
        desc = QLabel(
            "각 체질 카드 안의 문항을 직접 체크하세요. 여러 문항을 동시에 선택할 수 있으며, "
            "가장 많이 체크된 체질을 기본 체질 후보로 사용한 뒤 PSL_Test와 Face_AI 결과를 함께 반영해 최종 타입을 판단합니다."
        )
        desc.setObjectName("BodyText")
        desc.setWordWrap(True)
        survey_layout.addWidget(title)
        survey_layout.addWidget(desc)

        guide = QLabel(
            "설문 방식 안내\n"
            "• 체질별 4개 문항을 각각 개별 선택합니다.\n"
            "• 하나의 체질 안에서 여러 문항을 동시에 선택할 수 있습니다.\n"
            "• 선택 수가 많을수록 해당 체질 경향이 강한 것으로 반영됩니다."
        )
        guide.setObjectName("InfoBlock")
        guide.setWordWrap(True)
        survey_layout.addWidget(guide)

        survey_grid = QGridLayout()
        survey_grid.setSpacing(12)
        for idx, group in enumerate(SURVEY_GROUPS):
            card = QFrame()
            card.setObjectName("Panel")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(18, 18, 18, 18)
            card_layout.setSpacing(10)

            heading = QLabel(group["label"])
            heading.setObjectName("SectionTitle")
            sub = QLabel(group["subtitle"])
            sub.setObjectName("BodyText")
            sub.setWordWrap(True)

            item_count = QLabel(f"선택 항목: 0 / {len(group['items'])}")
            item_count.setObjectName("StatusText")
            self.survey_count_labels[group["key"]] = item_count

            checkbox_column = QWidget()
            checkbox_layout = QVBoxLayout(checkbox_column)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setSpacing(8)

            checkboxes: list[QCheckBox] = []
            for item in group["items"]:
                checkbox = QCheckBox(item)
                checkbox.toggled.connect(self.update_survey_preview)
                checkbox_layout.addWidget(checkbox)
                checkboxes.append(checkbox)

            self.survey_checkboxes[group["key"]] = checkboxes

            card_layout.addWidget(heading)
            card_layout.addWidget(sub)
            card_layout.addWidget(item_count)
            card_layout.addWidget(checkbox_column)
            survey_grid.addWidget(card, idx // 2, idx % 2)
        survey_layout.addLayout(survey_grid)

        self.survey_result_label = QLabel("기본 체질 미리보기: 계산 중")
        self.survey_result_label.setObjectName("StatusText")
        self.survey_result_label.setWordWrap(True)
        survey_layout.addWidget(self.survey_result_label)

        self.survey_summary_text = QPlainTextEdit()
        self.survey_summary_text.setReadOnly(True)
        self.survey_summary_text.setMinimumHeight(180)
        survey_layout.addWidget(self.wrap_group("설문 요약", self.survey_summary_text))
        layout.addWidget(survey_panel)

        action_grid = QGridLayout()
        action_grid.setSpacing(10)
        self.survey_back_button = QPushButton("시작 페이지로 돌아가기")
        self.survey_back_button.clicked.connect(lambda: self.go_to_step(0))
        self.survey_next_button = QPushButton("설문 완료 후 PSL_Test로 이동")
        self.survey_next_button.clicked.connect(self.complete_survey_and_continue)
        action_grid.addWidget(self.survey_back_button, 0, 0)
        action_grid.addWidget(self.survey_next_button, 0, 1)
        layout.addLayout(action_grid)
        layout.addStretch(1)
        return page

    def build_psl_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        config_panel = QFrame()
        config_panel.setObjectName("Panel")
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(18, 18, 18, 18)
        config_layout.setSpacing(12)

        title = QLabel("3. PSL_Test 측정")
        title.setObjectName("SectionTitle")
        desc = QLabel(
            "고정 Arduino baud rate와 고정 샘플레이트 기준으로 iPPG 단독 또는 카메라 + PPG 측정을 수행합니다."
        )
        desc.setObjectName("BodyText")
        desc.setWordWrap(True)
        config_layout.addWidget(title)
        config_layout.addWidget(desc)

        form_group = QGroupBox("측정 설정")
        form = QFormLayout(form_group)
        form.setLabelAlignment(ALIGN_TOP)

        self.psl_mode_combo = QComboBox()
        self.psl_mode_combo.addItem("PPG 단독", "ppg")
        self.psl_mode_combo.addItem("카메라 + PPG", "multimodal")

        self.psl_port_combo = QComboBox()
        self.psl_port_refresh = QPushButton("포트 새로고침")
        self.psl_port_refresh.clicked.connect(self.refresh_ports)
        port_row = QWidget()
        port_layout = QHBoxLayout(port_row)
        port_layout.setContentsMargins(0, 0, 0, 0)
        port_layout.setSpacing(10)
        port_layout.addWidget(self.psl_port_combo, 1)
        port_layout.addWidget(self.psl_port_refresh)

        self.psl_camera_combo = QComboBox()
        self.psl_camera_refresh = QPushButton("카메라 새로고침")
        self.psl_camera_refresh.clicked.connect(self.refresh_psl_cameras)
        psl_cam_row = QWidget()
        psl_cam_layout = QHBoxLayout(psl_cam_row)
        psl_cam_layout.setContentsMargins(0, 0, 0, 0)
        psl_cam_layout.setSpacing(10)
        psl_cam_layout.addWidget(self.psl_camera_combo, 1)
        psl_cam_layout.addWidget(self.psl_camera_refresh)

        self.psl_duration_spin = QDoubleSpinBox()
        self.psl_duration_spin.setRange(5.0, 300.0)
        self.psl_duration_spin.setValue(60.0)
        self.psl_duration_spin.setSuffix(" s")

        self.psl_age_spin = QSpinBox()
        self.psl_age_spin.setRange(0, 120)
        self.psl_age_spin.setSpecialValueText("선택 입력")

        self.psl_sex_combo = QComboBox()
        self.psl_sex_combo.addItem("미입력", "unknown")
        self.psl_sex_combo.addItem("남성", "male")
        self.psl_sex_combo.addItem("여성", "female")

        self.psl_sbp_input = QLineEdit()
        self.psl_sbp_input.setPlaceholderText("선택 입력 예) 118")
        self.psl_dbp_input = QLineEdit()
        self.psl_dbp_input.setPlaceholderText("선택 입력 예) 76")

        form.addRow("측정 모드", self.psl_mode_combo)
        form.addRow("시리얼 포트", port_row)
        form.addRow("카메라", psl_cam_row)
        form.addRow("Baud", QLabel("1000000 (고정)"))
        form.addRow("샘플레이트", QLabel("250 Hz (고정)"))
        form.addRow("측정 시간", self.psl_duration_spin)
        form.addRow("나이", self.psl_age_spin)
        form.addRow("성별", self.psl_sex_combo)
        form.addRow("보정 SBP", self.psl_sbp_input)
        form.addRow("보정 DBP", self.psl_dbp_input)
        config_layout.addWidget(form_group)

        button_grid = QGridLayout()
        button_grid.setSpacing(10)
        self.psl_start_button = QPushButton("PSL_Test 실행")
        self.psl_start_button.clicked.connect(self.start_psl_measurement)
        self.psl_to_face_button = QPushButton("얼굴 촬영으로 이동")
        self.psl_to_face_button.setEnabled(False)
        self.psl_to_face_button.clicked.connect(lambda: self.go_to_step(3))
        button_grid.addWidget(self.psl_start_button, 0, 0)
        button_grid.addWidget(self.psl_to_face_button, 0, 1)
        config_layout.addLayout(button_grid)

        self.psl_status_label = QLabel("PSL_Test 측정을 기다리는 중입니다.")
        self.psl_status_label.setObjectName("StatusText")
        self.psl_status_label.setWordWrap(True)
        config_layout.addWidget(self.psl_status_label)
        layout.addWidget(config_panel)

        psl_card_group = QGroupBox("PSL 실시간 요약")
        psl_card_layout = QGridLayout(psl_card_group)
        psl_card_meta = {
            "heart_rate": ("심박수", "#18c3ff"),
            "hrv": ("HRV", "#ff6b6b"),
            "stress": ("스트레스", "#f7a23b"),
            "blood_pressure": ("혈압 추정", "#88f1ff"),
        }
        for idx, (key, meta) in enumerate(psl_card_meta.items()):
            card = DashboardCard(meta[0], meta[1])
            self.psl_cards[key] = card
            psl_card_layout.addWidget(card, idx // 2, idx % 2)
        layout.addWidget(psl_card_group)

        self.psl_summary = QPlainTextEdit()
        self.psl_summary.setReadOnly(True)
        self.psl_summary.setMinimumHeight(220)
        self.psl_log = QPlainTextEdit()
        self.psl_log.setReadOnly(True)
        self.psl_log.setMinimumHeight(180)
        layout.addWidget(self.wrap_group("PSL 요약", self.psl_summary))
        layout.addWidget(self.wrap_group("PSL 로그", self.psl_log))
        return page

    def build_face_capture_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        capture_panel = QFrame()
        capture_panel.setObjectName("Panel")
        capture_layout = QVBoxLayout(capture_panel)
        capture_layout.setContentsMargins(18, 18, 18, 18)
        capture_layout.setSpacing(12)

        title = QLabel("4. 얼굴 촬영")
        title.setObjectName("SectionTitle")
        desc = QLabel(
            "실시간 미리보기를 시작하고 얼굴을 중앙에 맞춘 뒤 현재 프레임을 분석합니다. "
            "실제 Face_AI 분석 결과 확인은 다음 페이지에서 진행됩니다."
        )
        desc.setObjectName("BodyText")
        desc.setWordWrap(True)
        capture_layout.addWidget(title)
        capture_layout.addWidget(desc)

        control_group = QGroupBox("얼굴 촬영 설정")
        control_form = QFormLayout(control_group)
        control_form.setLabelAlignment(ALIGN_TOP)

        self.face_camera_combo = QComboBox()
        self.face_camera_refresh = QPushButton("카메라 새로고침")
        self.face_camera_refresh.clicked.connect(self.refresh_face_cameras)
        face_cam_row = QWidget()
        face_cam_layout = QHBoxLayout(face_cam_row)
        face_cam_layout.setContentsMargins(0, 0, 0, 0)
        face_cam_layout.setSpacing(10)
        face_cam_layout.addWidget(self.face_camera_combo, 1)
        face_cam_layout.addWidget(self.face_camera_refresh)

        face_button_grid = QGridLayout()
        face_button_grid.setSpacing(10)
        self.face_preview_button = QPushButton("미리보기 시작")
        self.face_preview_button.clicked.connect(self.restart_face_preview)
        self.face_analyze_button = QPushButton("현재 프레임 분석")
        self.face_analyze_button.clicked.connect(self.start_face_analysis)
        self.face_to_review_button = QPushButton("얼굴 분석 확인으로 이동")
        self.face_to_review_button.setEnabled(False)
        self.face_to_review_button.clicked.connect(lambda: self.go_to_step(4))
        face_button_grid.addWidget(self.face_preview_button, 0, 0)
        face_button_grid.addWidget(self.face_analyze_button, 0, 1)
        face_button_grid.addWidget(self.face_to_review_button, 1, 0, 1, 2)

        control_form.addRow("카메라", face_cam_row)
        control_form.addRow("동작", face_button_grid)
        capture_layout.addWidget(control_group)

        self.face_capture_status_label = QLabel("실시간 얼굴 미리보기를 시작하기 전입니다.")
        self.face_capture_status_label.setObjectName("StatusText")
        self.face_capture_status_label.setWordWrap(True)
        capture_layout.addWidget(self.face_capture_status_label)

        self.face_preview_label = AspectRatioLabel("여기에 실시간 얼굴 미리보기가 표시됩니다.")
        self.face_preview_label.setMinimumHeight(500)
        capture_layout.addWidget(self.face_preview_label)
        layout.addWidget(capture_panel)

        self.face_log = QPlainTextEdit()
        self.face_log.setReadOnly(True)
        self.face_log.setMinimumHeight(220)
        layout.addWidget(self.wrap_group("얼굴 촬영 로그", self.face_log))
        return page

    def build_face_review_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        review_panel = QFrame()
        review_panel.setObjectName("Panel")
        review_layout = QVBoxLayout(review_panel)
        review_layout.setContentsMargins(18, 18, 18, 18)
        review_layout.setSpacing(12)

        title = QLabel("5. Face_AI 분석 확인")
        title.setObjectName("SectionTitle")
        desc = QLabel(
            "여기에서 주석이 포함된 스냅샷과 Face_AI 피부 지표를 확인합니다. "
            "모든 통합 결과는 최종 결과 페이지에서만 한 번에 표시됩니다."
        )
        desc.setObjectName("BodyText")
        desc.setWordWrap(True)
        review_layout.addWidget(title)
        review_layout.addWidget(desc)

        review_button_grid = QGridLayout()
        review_button_grid.setSpacing(10)
        self.face_retake_button = QPushButton("얼굴 다시 촬영")
        self.face_retake_button.clicked.connect(lambda: self.go_to_step(3))
        self.face_to_final_button = QPushButton("최종 결과 열기")
        self.face_to_final_button.setEnabled(False)
        self.face_to_final_button.clicked.connect(lambda: self.go_to_step(5))
        review_button_grid.addWidget(self.face_retake_button, 0, 0)
        review_button_grid.addWidget(self.face_to_final_button, 0, 1)
        review_layout.addLayout(review_button_grid)

        self.face_review_status_label = QLabel("아직 Face_AI 결과가 없습니다.")
        self.face_review_status_label.setObjectName("StatusText")
        self.face_review_status_label.setWordWrap(True)
        review_layout.addWidget(self.face_review_status_label)

        self.face_snapshot_label = AspectRatioLabel("여기에 Face_AI 분석 스냅샷이 표시됩니다.")
        self.face_snapshot_label.setMinimumHeight(480)
        review_layout.addWidget(self.face_snapshot_label)
        layout.addWidget(review_panel)

        face_card_group = QGroupBox("Face_AI 지표")
        face_card_layout = QGridLayout(face_card_group)
        face_card_meta = {
            "overall": ("종합 점수", "#49e1b6"),
            "wrinkle": ("주름", "#ff7d5b"),
            "pigmentation": ("색소", "#f7c65a"),
            "pore": ("모공", "#49b6ff"),
            "dryness": ("건조", "#ff9db3"),
            "sagging": ("처짐", "#a2f56f"),
        }
        for idx, (key, meta) in enumerate(face_card_meta.items()):
            card = DashboardCard(meta[0], meta[1])
            self.face_review_cards[key] = card
            face_card_layout.addWidget(card, idx // 2, idx % 2)
        layout.addWidget(face_card_group)

        self.face_summary = QPlainTextEdit()
        self.face_summary.setReadOnly(True)
        self.face_summary.setMinimumHeight(240)
        layout.addWidget(self.wrap_group("Face_AI 요약", self.face_summary))
        return page

    def build_final_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        final_panel = QFrame()
        final_panel.setObjectName("Panel")
        final_layout = QVBoxLayout(final_panel)
        final_layout.setContentsMargins(18, 18, 18, 18)
        final_layout.setSpacing(14)

        title = QLabel("6. 최종 통합 결과")
        title.setObjectName("SectionTitle")
        desc = QLabel(
            "샘플 리포트 형태로 심혈관·피부·5장6부·체질 결과를 한 장의 보고서 흐름으로 정리했습니다."
        )
        desc.setObjectName("BodyText")
        desc.setWordWrap(True)
        final_layout.addWidget(title)
        final_layout.addWidget(desc)

        action_grid = QGridLayout()
        action_grid.setSpacing(10)
        self.summary_export_button = QPushButton("결과 저장")
        self.summary_export_button.clicked.connect(self.export_combined_summary)
        self.summary_folder_button = QPushButton("세션 폴더 열기")
        self.summary_folder_button.clicked.connect(self.open_session_dir)
        self.summary_restart_button = QPushButton("세션 다시 시작")
        self.summary_restart_button.clicked.connect(self.start_new_session)
        action_grid.addWidget(self.summary_export_button, 0, 0)
        action_grid.addWidget(self.summary_folder_button, 0, 1)
        action_grid.addWidget(self.summary_restart_button, 1, 0, 1, 2)
        final_layout.addLayout(action_grid)

        self.final_status_label = QLabel("이 페이지를 채우려면 PSL_Test와 Face_AI 단계를 완료하세요.")
        self.final_status_label.setObjectName("StatusText")
        self.final_status_label.setWordWrap(True)
        final_layout.addWidget(self.final_status_label)
        layout.addWidget(final_panel)

        self.final_report_sections = []

        self.final_biosignal_section = ReportSection(
            "1.심혈관·자율신경 영역 측정 및 AI 분석 리포트",
            "#49d9ff",
            "PSL_Test 기반 측정값, 정상 범위, AI 해석, 고객 설명 자료를 한 번에 정리합니다.",
        )
        self.final_biosignal_table = ReportTableWidget("#49d9ff")
        self.final_biosignal_section.add_content(self.final_biosignal_table)
        self.final_report_sections.append(self.final_biosignal_section)
        layout.addWidget(self.final_biosignal_section)

        self.final_skin_section = ReportSection(
            "2. 피부·미용 영역 측정 및 AI 분석 리포트",
            "#f7b557",
            "Face_AI 지표를 피부 상태 평가와 관리 방향 중심으로 보기 쉽게 정리합니다.",
        )
        self.final_skin_table = ReportTableWidget("#f7b557")
        self.final_skin_section.add_content(self.final_skin_table)
        self.final_report_sections.append(self.final_skin_section)
        layout.addWidget(self.final_skin_section)

        self.final_organ_section = ReportSection(
            "3. 5장 6부 건강 상태 추정 AI 분석 리포트",
            "#7ef0b5",
            "(질병 진단이 아니라 경향성 / 밸런스 점수 상태)",
        )
        self.final_zang_table = ReportTableWidget("#7ef0b5")
        self.final_fu_table = ReportTableWidget("#9ddfff")
        self.final_organ_section.add_content(self.final_zang_table)
        self.final_organ_section.add_content(self.final_fu_table)
        self.final_report_sections.append(self.final_organ_section)
        layout.addWidget(self.final_organ_section)

        self.final_constitution_section = ReportSection(
            "4. 체질 AI 분석 리포트",
            "#ff8f7a",
            "체질 설문, PSL_Test, Face_AI 결과를 결합해 8개 타입 중 최종 판정을 표시합니다.",
        )
        self.final_constitution_table = ReportTableWidget("#ff8f7a")
        self.final_constitution_section.add_content(self.final_constitution_table)
        self.final_report_sections.append(self.final_constitution_section)
        layout.addWidget(self.final_constitution_section)

        self.recommendation_text = QPlainTextEdit()
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setMinimumHeight(240)
        layout.addWidget(self.wrap_group("추가 판정 해설", self.recommendation_text))

        self.final_summary_text = QPlainTextEdit()
        self.final_summary_text.setReadOnly(True)
        self.final_summary_text.setMinimumHeight(260)
        self.final_paths_text = QPlainTextEdit()
        self.final_paths_text.setReadOnly(True)
        self.final_paths_text.setMinimumHeight(160)
        layout.addWidget(self.wrap_group("통합 세션 요약", self.final_summary_text))
        layout.addWidget(self.wrap_group("저장된 파일", self.final_paths_text))
        self.populate_final_report_placeholders()
        return page

    def wrap_group(self, title: str, widget: QWidget) -> QGroupBox:
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(12, 12, 12, 12)
        group_layout.addWidget(widget)
        return group

    def current_reference_age(self) -> int | None:
        if not hasattr(self, "psl_age_spin"):
            return None
        age_value = int(self.psl_age_spin.value())
        return age_value if age_value > 0 else None

    def clamp_metric(self, value: float, low: float = 0.0, high: float = 100.0) -> float:
        return max(low, min(high, float(value)))

    def face_metric_score(self, metric_name: str, default: float = 0.0) -> float:
        metrics = (self.face_result or {}).get("metrics") or {}
        metric = metrics.get(metric_name) or {}
        return safe_float(metric.get("score"), default)

    def describe_balance_state(self, score: float) -> str:
        if score >= 82.0:
            return "매우 안정"
        if score >= 72.0:
            return "양호"
        if score >= 60.0:
            return "보통"
        if score >= 48.0:
            return "약간 불균형"
        return "관리 필요"

    def placeholder_rows(self, labels: list[str], column_count: int, detail: str = "측정 대기 중") -> list[list[str]]:
        return [[label, detail] + ["-"] * max(0, column_count - 2) for label in labels]

    def build_biosignal_report_rows(self) -> list[list[str]]:
        headers_count = 5
        if not self.psl_report:
            return self.placeholder_rows(
                ["심박수 (HR)", "HRV (심박변이도)", "혈류 순환 지수", "혈관 건강 지수", "스트레스 지수", "혈관 나이", "혈압 추정"],
                headers_count,
            )

        heart = self.psl_report.get("heart_rate") or {}
        hrv = self.psl_report.get("hrv") or {}
        stress = self.psl_report.get("stress") or {}
        circulation = self.psl_report.get("circulation") or {}
        vascular_health = self.psl_report.get("vascular_health") or {}
        vascular_age = self.psl_report.get("vascular_age") or {}
        blood_pressure = self.psl_report.get("blood_pressure") or {}

        heart_rate = safe_float(heart.get("heart_rate_bpm"), 0.0)
        rmssd = safe_float(hrv.get("rmssd_ms"), 0.0)
        circulation_score = safe_float(circulation.get("circulation_score"), 0.0)
        perfusion = safe_float(circulation.get("perfusion_index"), 0.0)
        vascular_score = safe_float(vascular_health.get("vascular_health_score"), 0.0)
        stress_score = safe_float(stress.get("stress_score"), 0.0)
        estimated_age = safe_float(vascular_age.get("vascular_age_estimate"), 0.0)
        age_gap = safe_float(vascular_age.get("vascular_age_gap"), 0.0)
        sbp = safe_float(blood_pressure.get("estimated_sbp"), 0.0)
        dbp = safe_float(blood_pressure.get("estimated_dbp"), 0.0)
        reference_age = self.current_reference_age()

        if heart_rate <= 0.0:
            hr_ai = "측정 신호 부족"
            hr_guide = "센서 접촉과 조명을 다시 확인한 뒤 재측정하는 것이 좋습니다."
        elif heart_rate < 60.0:
            hr_ai = "낮은 심박 경향"
            hr_guide = "편안한 상태일 수 있으나 어지럼·피로가 있으면 추가 확인이 필요합니다."
        elif heart_rate <= 100.0:
            hr_ai = "정상 범위"
            hr_guide = "현재 심장 박동은 안정적인 상태로 해석됩니다."
        elif heart_rate <= 110.0:
            hr_ai = "약간 빠른 상태"
            hr_guide = "긴장, 움직임, 카페인 섭취 등 일시적인 요인이 반영됐을 수 있습니다."
        else:
            hr_ai = "높은 심박 경향"
            hr_guide = "스트레스나 피로가 누적된 상태일 수 있어 휴식 후 재확인이 좋습니다."

        if rmssd <= 0.0:
            hrv_ai = "측정 신호 부족"
            hrv_guide = "HRV는 안정된 자세와 충분한 측정 시간이 있을 때 더 정확합니다."
        elif 40.0 <= rmssd <= 80.0:
            hrv_ai = "자율신경 균형 양호"
            hrv_guide = "스트레스 대응과 회복 여유가 비교적 좋은 편으로 해석됩니다."
        elif rmssd < 25.0:
            hrv_ai = "회복 여유 낮음"
            hrv_guide = "피로가 누적됐을 가능성이 있어 수면과 휴식 확보가 중요합니다."
        else:
            hrv_ai = "회복 리듬 보통"
            hrv_guide = "자율신경 균형은 유지되지만 컨디션에 따라 변동성이 있을 수 있습니다."

        if circulation_score >= 75.0:
            circulation_ai = "순환 우수"
            circulation_guide = "말초 혈류순환과 관류 흐름이 원활한 상태로 보입니다."
        elif circulation_score >= 60.0:
            circulation_ai = "순환 양호"
            circulation_guide = "전반적인 혈류 흐름은 무난한 편입니다."
        elif circulation_score >= 45.0:
            circulation_ai = "순환 보통"
            circulation_guide = "순환이 크게 나쁘진 않지만 온열·유산소 관리가 도움될 수 있습니다."
        else:
            circulation_ai = "순환 저하 경향"
            circulation_guide = "혈류순환이 둔해진 패턴이 보여 활동량과 온열 관리가 권장됩니다."

        if vascular_score >= 75.0:
            vascular_ai = "혈관 탄력 우수"
            vascular_guide = "혈관 탄성과 반사 지표가 안정적인 편입니다."
        elif vascular_score >= 65.0:
            vascular_ai = "혈관 탄력 양호"
            vascular_guide = "혈관 상태는 정상 범위 안에서 잘 유지되고 있습니다."
        elif vascular_score >= 50.0:
            vascular_ai = "관리 필요 보통"
            vascular_guide = "생활 리듬과 유산소 운동을 통해 탄력 유지가 필요합니다."
        else:
            vascular_ai = "탄력 관리 필요"
            vascular_guide = "혈관 탄성 관련 지표가 낮아 혈류·운동 관리가 권장됩니다."

        if stress_score < 35.0:
            stress_ai = "안정 상태"
            stress_guide = "현재 긴장도는 비교적 낮고 안정적인 상태로 해석됩니다."
        elif stress_score <= 60.0:
            stress_ai = "관리 가능 범위"
            stress_guide = "일상 스트레스 범위 안이지만 회복 루틴을 유지하는 것이 좋습니다."
        elif stress_score <= 75.0:
            stress_ai = "약간 높은 상태"
            stress_guide = "긴장이나 피로가 누적됐을 수 있어 호흡·휴식 관리가 필요합니다."
        else:
            stress_ai = "높은 긴장 상태"
            stress_guide = "자율신경 긴장도가 높아 휴식과 수면 관리가 우선입니다."

        if reference_age is not None:
            age_range = f"{reference_age}세 ±5세"
            age_ai = f"실제보다 {age_gap:+.1f}세"
            if age_gap >= 3.0:
                age_guide = "혈관 건강 관리를 조금 더 강화하는 것이 좋습니다."
            elif age_gap <= -3.0:
                age_guide = "혈류와 탄력 관리가 비교적 잘 유지되고 있습니다."
            else:
                age_guide = "현재 혈관 연령은 실제 나이와 비슷한 수준으로 보입니다."
        else:
            age_range = "실제 나이 ±5세"
            age_ai = "나이 기준 미입력"
            age_guide = "입력한 나이가 없어서 추정 혈관 나이만 참고용으로 표시합니다."

        if sbp <= 0.0 or dbp <= 0.0:
            bp_ai = "측정 신호 부족"
            bp_guide = "혈압 추정은 보정값과 안정 측정 조건이 확보될수록 더 유용합니다."
        elif sbp <= 120.0 and dbp <= 80.0:
            bp_ai = "정상 범위"
            bp_guide = "현재 혈압 경향은 정상 범위로 해석됩니다."
        elif sbp <= 129.0 and dbp < 80.0:
            bp_ai = "정상 상한"
            bp_guide = "정상 범위 상단에 가까워 생활 습관 관리가 도움이 됩니다."
        elif sbp < 140.0 or dbp < 90.0:
            bp_ai = "경계 범위"
            bp_guide = "긴장도와 생활 습관에 따라 변동될 수 있어 반복 측정이 좋습니다."
        else:
            bp_ai = "높은 편"
            bp_guide = "반복 측정 시에도 높게 나오면 별도 혈압 확인이 필요합니다."

        return [
            ["심박수 (HR)", f"{heart_rate:.1f} bpm", "60~100 bpm", hr_ai, hr_guide],
            ["HRV (심박변이도)", f"RMSSD {rmssd:.1f} ms", "40~80 ms", hrv_ai, hrv_guide],
            ["혈류 순환 지수", f"{circulation_score:.1f} / 100", "60 이상", circulation_ai, f"{circulation_guide} (관류지수 {perfusion:.3f})"],
            ["혈관 건강 지수", f"{vascular_score:.1f} / 100", "65 이상", vascular_ai, vascular_guide],
            ["스트레스 지수", f"{stress_score:.1f} / 100", "40~60", stress_ai, stress_guide],
            ["혈관 나이", f"{estimated_age:.1f}세", age_range, age_ai, age_guide],
            ["혈압 추정", f"{sbp:.0f} / {dbp:.0f} mmHg", "120 / 80 기준", bp_ai, bp_guide],
        ]

    def build_skin_report_rows(self) -> list[list[str]]:
        headers_count = 5
        row_titles = ["홍반 / 붉은기", "색소 / 톤", "모공", "주름", "유분-건조 경향", "여드름", "잡티"]
        if not self.face_result or not self.face_result.get("face_detected", True):
            missing_detail = "측정 대기 중" if not self.face_result else "얼굴 감지 필요"
            return self.placeholder_rows(row_titles, headers_count, missing_detail)

        stress_score = safe_float(((self.psl_report or {}).get("stress") or {}).get("stress_score"), 42.0)
        heart_rate = safe_float(((self.psl_report or {}).get("heart_rate") or {}).get("heart_rate_bpm"), 72.0)
        circulation_score = safe_float(((self.psl_report or {}).get("circulation") or {}).get("circulation_score"), 55.0)
        overall_score = safe_float((self.face_result or {}).get("overall_score"), 50.0)

        pigmentation = self.face_metric_score("pigmentation", 0.0)
        pore = self.face_metric_score("pore", 0.0)
        wrinkle = self.face_metric_score("wrinkle", 0.0)
        dryness = self.face_metric_score("dryness", 0.0)
        sagging = self.face_metric_score("sagging", 0.0)

        redness_level = self.clamp_metric(0.52 * stress_score + 0.18 * max(0.0, heart_rate - 70.0) + 0.15 * (100.0 - circulation_score) + 0.15 * overall_score)
        redness_control = self.clamp_metric(100.0 - redness_level)
        tone_balance = self.clamp_metric(100.0 - pigmentation * 0.72 - redness_level * 0.18)
        pore_balance = self.clamp_metric(100.0 - pore * 0.75)
        elasticity = self.clamp_metric(100.0 - (0.65 * wrinkle + 0.35 * sagging))
        oil_level = self.clamp_metric(44.0 + pore * 0.35 - dryness * 0.22 + redness_level * 0.12)
        water_level = self.clamp_metric(64.0 - dryness * 0.48 - redness_level * 0.16 + circulation_score * 0.12)
        moisture_balance = self.clamp_metric(100.0 - abs(oil_level - water_level) * 1.6 - dryness * 0.25)
        trouble_risk = self.clamp_metric(0.45 * redness_level + 0.35 * pore + 0.20 * pigmentation)
        trouble_stability = self.clamp_metric(100.0 - trouble_risk)
        spot_area = self.clamp_metric(pigmentation * 0.26 + overall_score * 0.08)
        clarity = self.clamp_metric(100.0 - pigmentation * 0.65 - redness_level * 0.12)

        if redness_level < 25.0:
            redness_state = "안정 피부"
            redness_direction = "진정 루틴 유지"
        elif redness_level < 45.0:
            redness_state = "약간 민감"
            redness_direction = "쿨링·진정 관리 추천"
        elif redness_level < 65.0:
            redness_state = "열감 주의"
            redness_direction = "자극 완화 / 냉각 관리"
        else:
            redness_state = "붉은기 관리 필요"
            redness_direction = "열 진정 중심 관리"

        if tone_balance >= 75.0:
            tone_state = "톤 균형 양호"
            tone_direction = "현재 톤 관리 유지"
        elif tone_balance >= 60.0:
            tone_state = "톤 편차 약간"
            tone_direction = "미백·톤 개선 관리"
        else:
            tone_state = "색소 불균형"
            tone_direction = "색소·자외선 관리 강화"

        if pore_balance >= 70.0:
            pore_state = "모공 정돈 양호"
            pore_direction = "유수분 균형 유지"
        elif pore_balance >= 55.0:
            pore_state = "모공 약간 확대"
            pore_direction = "피지관리 + 탄력 케어"
        else:
            pore_state = "모공 관리 필요"
            pore_direction = "피지 억제 / 리프팅 관리"

        if elasticity >= 72.0:
            wrinkle_state = "탄력 양호"
            wrinkle_direction = "예방 중심 관리"
        elif elasticity >= 58.0:
            wrinkle_state = "초기 주름"
            wrinkle_direction = "콜라겐·재생 관리"
        else:
            wrinkle_state = "탄력 저하"
            wrinkle_direction = "재생·탄력 집중 관리"

        if dryness >= 65.0:
            sebum_state = "건조 우세"
            sebum_direction = "보습 중심 관리"
        elif oil_level - water_level >= 10.0:
            sebum_state = "유분 우세"
            sebum_direction = "피지 밸런스 조절"
        elif water_level - oil_level >= 10.0:
            sebum_state = "수분 우세"
            sebum_direction = "장벽 유지 중심 관리"
        else:
            sebum_state = "유수분 균형형"
            sebum_direction = "균형 유지 관리"

        if trouble_risk < 25.0:
            acne_state = "트러블 안정"
            acne_direction = "예방 위주 관리"
        elif trouble_risk < 45.0:
            acne_state = "가벼운 주의"
            acne_direction = "진정·피지 관리"
        else:
            acne_state = "트러블 가능성"
            acne_direction = "염증 완화 / 자극 최소화"

        if clarity >= 72.0:
            spot_state = "맑기 양호"
            spot_direction = "광채 유지 관리"
        elif clarity >= 58.0:
            spot_state = "잡티 경향 약간"
            spot_direction = "톤·광채 관리"
        else:
            spot_state = "잡티 관리 필요"
            spot_direction = "색소·광채 집중 관리"

        return [
            ["홍반 / 붉은기", f"홍조 반응 {redness_level:.0f}%", f"진정 지수 {redness_control:.0f}", redness_state, redness_direction],
            ["색소 / 톤", f"색소 밀도 {pigmentation * 0.32:.0f}%", f"톤 균형 {tone_balance:.0f}", tone_state, tone_direction],
            ["모공", f"평균 모공 크기 {0.18 + pore * 0.004:.2f} mm", f"모공 정돈 {pore_balance:.0f}", pore_state, pore_direction],
            ["주름", f"주름 깊이 {0.08 + wrinkle * 0.0038:.2f} mm", f"탄력 지수 {elasticity:.0f}", wrinkle_state, wrinkle_direction],
            ["유분-건조 경향", f"유분 {oil_level:.0f}% / 수분 {water_level:.0f}%", f"수분 밸런스 {moisture_balance:.0f}", sebum_state, sebum_direction],
            ["여드름", f"트러블 위험 {trouble_risk:.0f}%", f"안정 지수 {trouble_stability:.0f}", acne_state, acne_direction],
            ["잡티", f"잡티 면적 {spot_area:.0f}%", f"맑기 지수 {clarity:.0f}", spot_state, spot_direction],
        ]

    def build_organ_balance_rows(self) -> tuple[list[list[str]], list[list[str]]]:
        headers_count = 5
        five_labels = ["간(肝) 계통", "심(心) 계통", "비(脾) 계통", "폐(肺) 계통", "신(腎) 계통"]
        six_labels = ["위", "대장", "소장", "방광", "담", "삼초"]
        if not self.psl_report and not self.face_result:
            return self.placeholder_rows(five_labels, headers_count), self.placeholder_rows(six_labels, headers_count)

        heart = (self.psl_report or {}).get("heart_rate") or {}
        hrv = (self.psl_report or {}).get("hrv") or {}
        stress = (self.psl_report or {}).get("stress") or {}
        circulation = (self.psl_report or {}).get("circulation") or {}
        vascular_health = (self.psl_report or {}).get("vascular_health") or {}
        vascular_age = (self.psl_report or {}).get("vascular_age") or {}
        blood_pressure = (self.psl_report or {}).get("blood_pressure") or {}

        hr = safe_float(heart.get("heart_rate_bpm"), 72.0)
        rmssd = safe_float(hrv.get("rmssd_ms"), 30.0)
        sdnn = safe_float(hrv.get("sdnn_ms"), 40.0)
        hrv_score = safe_float(hrv.get("hrv_score"), 0.0)
        if hrv_score <= 0.0:
            hrv_score = self.clamp_metric((rmssd - 10.0) * 1.35 + (sdnn - 15.0) * 0.55)
        stress_score = safe_float(stress.get("stress_score"), 45.0)
        circulation_score = safe_float(circulation.get("circulation_score"), 55.0)
        vascular_score = safe_float(vascular_health.get("vascular_health_score"), 55.0)
        sbp = safe_float(blood_pressure.get("estimated_sbp"), 120.0)
        dbp = safe_float(blood_pressure.get("estimated_dbp"), 80.0)
        age_gap = abs(safe_float(vascular_age.get("vascular_age_gap"), 0.0))

        pigmentation = self.face_metric_score("pigmentation", 40.0 if self.face_result else 0.0)
        pore = self.face_metric_score("pore", 40.0 if self.face_result else 0.0)
        wrinkle = self.face_metric_score("wrinkle", 35.0 if self.face_result else 0.0)
        dryness = self.face_metric_score("dryness", 35.0 if self.face_result else 0.0)
        sagging = self.face_metric_score("sagging", 35.0 if self.face_result else 0.0)

        bp_balance = self.clamp_metric(100.0 - abs(sbp - 120.0) * 0.9 - abs(dbp - 80.0) * 1.1)
        age_balance = self.clamp_metric(100.0 - age_gap * 12.0)

        def composite(parts: list[tuple[float, float]]) -> float:
            return self.clamp_metric(sum(weight * value for weight, value in parts))

        zang_scores = {
            "간(肝) 계통": composite([(0.35, 100.0 - stress_score), (0.25, 100.0 - pigmentation), (0.20, hrv_score), (0.20, bp_balance)]),
            "심(心) 계통": composite([(0.35, bp_balance), (0.35, hrv_score), (0.30, self.clamp_metric(100.0 - abs(hr - 72.0) * 2.2))]),
            "비(脾) 계통": composite([(0.30, circulation_score), (0.30, 100.0 - dryness), (0.20, 100.0 - stress_score), (0.20, 100.0 - pore)]),
            "폐(肺) 계통": composite([(0.40, vascular_score), (0.25, circulation_score), (0.20, 100.0 - dryness), (0.15, 100.0 - wrinkle)]),
            "신(腎) 계통": composite([(0.35, hrv_score), (0.25, vascular_score), (0.20, age_balance), (0.20, 100.0 - dryness)]),
        }
        fu_scores = {
            "위": composite([(0.35, circulation_score), (0.25, 100.0 - stress_score), (0.20, 100.0 - dryness), (0.20, bp_balance)]),
            "대장": composite([(0.35, 100.0 - pigmentation), (0.25, circulation_score), (0.20, 100.0 - pore), (0.20, 100.0 - stress_score)]),
            "소장": composite([(0.30, hrv_score), (0.25, 100.0 - stress_score), (0.20, circulation_score), (0.25, 100.0 - pigmentation)]),
            "방광": composite([(0.35, 100.0 - dryness), (0.30, circulation_score), (0.20, vascular_score), (0.15, bp_balance)]),
            "담": composite([(0.30, 100.0 - stress_score), (0.20, vascular_score), (0.20, 100.0 - pore), (0.15, hrv_score), (0.15, 100.0 - pigmentation)]),
            "삼초": composite([(0.25, hrv_score), (0.25, circulation_score), (0.20, vascular_score), (0.15, 100.0 - stress_score), (0.15, 100.0 - sagging)]),
        }

        zang_guides = {
            "간(肝) 계통": ("긴장·피로 경향", "스트레스 리듬 양호", "휴식 / 간 해독 식습관", "현재 생활 리듬 유지"),
            "심(心) 계통": ("긴장·교감신경 항진", "순환·안정 양호", "호흡 훈련 / 카페인 조절", "안정 루틴 유지"),
            "비(脾) 계통": ("소화·에너지 저하", "흡수·에너지 균형 양호", "규칙적 식사 / 온식 중심", "현재 식사 리듬 유지"),
            "폐(肺) 계통": ("호흡·건조 경향", "호흡·수분 균형 양호", "유산소 운동 / 수분 보충", "호흡 루틴 유지"),
            "신(腎) 계통": ("회복력 저하", "회복 리듬 양호", "충분한 수면 / 무리한 자극 최소", "회복 루틴 유지"),
        }
        fu_guides = {
            "위": ("소화 리듬 저하", "소화 리듬 안정", "식사 간격 일정화", "현재 식사 리듬 유지"),
            "대장": ("배출·순환 저하", "배출 균형 양호", "식이섬유 / 수분 관리", "배출 루틴 유지"),
            "소장": ("흡수·밸런스 변동", "흡수 밸런스 양호", "과식 피하기 / 규칙적 식사", "현재 식사 패턴 유지"),
            "방광": ("수분 대사 저하", "수분 순환 양호", "수분 섭취 / 냉자극 줄이기", "수분 리듬 유지"),
            "담": ("결단·긴장 변동", "긴장 조절 양호", "짧은 스트레칭 / 루틴 관리", "현재 루틴 유지"),
            "삼초": ("전신 밸런스 분산", "전신 균형 양호", "활동·휴식 리듬 재정비", "현재 리듬 유지"),
        }

        def build_rows(scores: dict[str, float], guides: dict[str, tuple[str, str, str, str]]) -> list[list[str]]:
            rows: list[list[str]] = []
            for label, score in scores.items():
                low_meaning, high_meaning, low_advice, high_advice = guides[label]
                status = self.describe_balance_state(score)
                if score >= 70.0:
                    meaning = high_meaning
                    advice = high_advice
                elif score >= 58.0:
                    meaning = f"{low_meaning} / 회복 가능"
                    advice = "과한 자극 없이 현재 리듬을 조금 더 안정적으로 유지"
                else:
                    meaning = low_meaning
                    advice = low_advice
                rows.append([label, f"{score:.0f} / 100", status, meaning, advice])
            return rows

        return build_rows(zang_scores, zang_guides), build_rows(fu_scores, fu_guides)

    def build_constitution_report_rows(self) -> tuple[list[list[str]], set[int]]:
        profile_pairs = {
            "soyang": ["heat_soyang", "weak_soyang"],
            "taeeum": ["damp_taeeum", "dry_taeeum"],
            "soeum": ["cold_soeum", "weak_soeum"],
            "taeyang": ["solid_taeyang", "weak_taeyang"],
        }

        recommendation = self.survey_result
        if recommendation is None and sum(self.survey_answers.values()) > 0:
            recommendation = build_profile_recommendation(self.survey_answers, self.psl_report, self.face_result, self.survey_details)

        emphasized_rows: set[int] = set()
        rows: list[list[str]] = []
        for index, group in enumerate(SURVEY_GROUPS):
            group_key = group["key"]
            pair_keys = profile_pairs[group_key]
            pair_labels = "\n".join(PROFILE_LIBRARY[key]["label"] for key in pair_keys)
            score_count = int(self.survey_answers.get(group_key, 0))
            selected_items = self.survey_details.get(group_key, [])

            if recommendation and recommendation.get("constitution_key") == group_key:
                emphasized_rows.add(index)
                detail_parts = [str(recommendation.get("summary") or "-")]
                if selected_items:
                    detail_parts.append(f"선택 문항: {', '.join(selected_items[:3])}")
                detail_parts.append(f"관리 목표: {recommendation.get('goal') or '-'}")
                judgement = f"현재 판정\n{recommendation.get('profile_label') or '-'}"
            else:
                detail_parts = [
                    f"{PROFILE_LIBRARY[pair_keys[0]]['label'].split(' (')[0]}: {PROFILE_LIBRARY[pair_keys[0]]['features'][0]}",
                    f"{PROFILE_LIBRARY[pair_keys[1]]['label'].split(' (')[0]}: {PROFILE_LIBRARY[pair_keys[1]]['features'][0]}",
                ]
                if score_count > 0:
                    detail_parts.append(f"설문 선택 {score_count}개")
                judgement = "대기"

            rows.append(
                [
                    group["label"],
                    pair_labels,
                    "\n".join(detail_parts),
                    judgement,
                ]
            )
        return rows, emphasized_rows

    def populate_final_report_placeholders(self) -> None:
        self.final_biosignal_table.set_table(
            ["측정 항목", "측정 데이터", "정상 범위", "AI 분석 결과", "고객에게 설명 자료"],
            self.placeholder_rows(["심박수 (HR)", "HRV (심박변이도)", "혈류 순환 지수", "혈관 건강 지수", "스트레스 지수", "혈관 나이", "혈압 추정"], 5),
            column_stretches=[16, 16, 13, 18, 31],
        )
        self.final_skin_table.set_table(
            ["측정 항목", "측정 데이터", "AI 분석 지수", "피부 상태 평가", "추천 관리 방향"],
            self.placeholder_rows(["홍반 / 붉은기", "색소 / 톤", "모공", "주름", "유분-건조 경향", "여드름", "잡티"], 5),
            column_stretches=[16, 23, 16, 18, 27],
        )
        self.final_zang_table.set_table(
            ["5 장부 영역", "AI 밸런스 점수", "상태 평가", "의미", "생활 관리 제안"],
            self.placeholder_rows(["간(肝) 계통", "심(心) 계통", "비(脾) 계통", "폐(肺) 계통", "신(腎) 계통"], 5),
            caption="5장부 영역",
            column_stretches=[15, 18, 16, 22, 29],
        )
        self.final_fu_table.set_table(
            ["6 부 영역", "AI 밸런스 점수", "상태 평가", "의미", "생활 관리 제안"],
            self.placeholder_rows(["위", "대장", "소장", "방광", "담", "삼초"], 5),
            caption="6부 영역",
            column_stretches=[15, 18, 16, 22, 29],
        )
        constitution_rows, emphasized_rows = self.build_constitution_report_rows()
        self.final_constitution_table.set_table(
            ["기본 4성 체질", "8성 체질", "고객의 체질 특징 설명 (8개에서 1가지)", "체질 판별"],
            constitution_rows,
            column_stretches=[14, 22, 42, 18],
            emphasized_rows=emphasized_rows,
        )

    def animate_final_report_sections(self) -> None:
        self._final_report_animations = []
        for index, widget in enumerate(self.final_report_sections):
            effect = QGraphicsOpacityEffect(widget)
            effect.setOpacity(0.0)
            widget.setGraphicsEffect(effect)

            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(320)
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.OutCubic)

            def start_animation(anim=animation) -> None:
                anim.start()

            def cleanup(target=widget, opacity_effect=effect) -> None:
                if target.graphicsEffect() is opacity_effect:
                    target.setGraphicsEffect(None)

            animation.finished.connect(cleanup)
            self._final_report_animations.append((effect, animation))
            QTimer.singleShot(90 * index, start_animation)

    def apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                color: #f3fbff;
            }
            QFrame#Hero {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(18, 45, 70, 240),
                    stop: 0.55 rgba(14, 87, 118, 232),
                    stop: 1 rgba(247, 162, 59, 218)
                );
                border: 1px solid rgba(211, 244, 255, 0.2);
                border-radius: 28px;
            }
            QLabel#HeroTitle {
                color: #f8fcff;
                font-family: Bahnschrift;
                font-size: 34px;
                font-weight: 700;
                letter-spacing: 2px;
            }
            QLabel#HeroSubtitle {
                color: rgba(248, 252, 255, 0.92);
                font-size: 13px;
                line-height: 1.5;
            }
            QFrame#Panel, QGroupBox, QFrame#DashboardCard, QFrame#StepBadge {
                background: rgba(8, 22, 34, 226);
                border: 1px solid rgba(169, 227, 255, 0.14);
                border-radius: 24px;
            }
            QFrame#ReportSection {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(8, 24, 38, 0.98),
                    stop: 1 rgba(10, 31, 48, 0.98)
                );
                border: 1px solid rgba(169, 227, 255, 0.16);
                border-radius: 26px;
            }
            QFrame#ReportTableFrame {
                background: rgba(4, 15, 24, 0.94);
                border: 1px solid rgba(126, 205, 233, 0.18);
                border-radius: 18px;
            }
            QGroupBox {
                margin-top: 14px;
                padding-top: 16px;
                font-size: 12px;
                font-weight: 700;
                color: #9ddfff;
            }
            QLabel#SectionTitle {
                font-size: 24px;
                font-weight: 700;
                color: #f6fbff;
            }
            QLabel#ReportSectionTitle {
                font-size: 28px;
                font-weight: 700;
                color: #f8fcff;
            }
            QLabel#ReportSectionSubtitle {
                font-size: 12px;
                color: #bed9e8;
            }
            QLabel#ReportCaption {
                font-size: 11px;
                color: #9ddfff;
                padding-left: 4px;
            }
            QLabel#BodyText, QLabel#InfoText, QLabel#StatusText {
                font-size: 12px;
                color: #c7e6f4;
            }
            QLabel#InfoBlock {
                background: rgba(13, 49, 69, 0.84);
                border: 1px solid rgba(247, 162, 59, 0.28);
                border-radius: 18px;
                padding: 16px;
                color: #ffe8c7;
            }
            QPushButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(24, 145, 196, 220),
                    stop: 1 rgba(70, 218, 198, 220)
                );
                color: #031117;
                border: 0;
                border-radius: 18px;
                padding: 14px 18px;
                min-height: 58px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(58, 193, 243, 230),
                    stop: 1 rgba(102, 237, 214, 230)
                );
            }
            QPushButton:pressed {
                background: rgba(77, 224, 226, 180);
            }
            QPushButton:disabled {
                background: rgba(102, 134, 148, 0.45);
                color: rgba(244, 250, 255, 0.6);
            }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background: rgba(3, 16, 26, 0.92);
                border: 1px solid rgba(157, 223, 255, 0.14);
                border-radius: 16px;
                padding: 10px 12px;
                color: #f3fbff;
                selection-background-color: #18c3ff;
            }
            QCheckBox {
                spacing: 10px;
                font-size: 12px;
                color: #e8f7ff;
                padding: 7px 8px;
                border-radius: 14px;
                background: rgba(12, 35, 49, 0.78);
                border: 1px solid rgba(90, 177, 211, 0.18);
            }
            QCheckBox:hover {
                background: rgba(17, 50, 70, 0.88);
                border-color: rgba(144, 229, 255, 0.3);
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QCheckBox::indicator:unchecked {
                border-radius: 6px;
                border: 1px solid rgba(157, 223, 255, 0.34);
                background: rgba(2, 12, 21, 0.96);
            }
            QCheckBox::indicator:checked {
                border-radius: 6px;
                border: 1px solid rgba(117, 245, 219, 0.55);
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(24, 195, 255, 0.95),
                    stop: 1 rgba(73, 225, 182, 0.95)
                );
            }
            QPlainTextEdit {
                padding: 14px;
                font-size: 12px;
                line-height: 1.4;
            }
            QLabel#CardTitle {
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
            }
            QLabel#CardValue {
                font-size: 25px;
                font-weight: 700;
                color: #f8fcff;
            }
            QLabel#CardDetail {
                font-size: 11px;
                color: #a8cbde;
            }
            QFrame#DashboardCard {
                border: 1px solid rgba(155, 231, 255, 0.16);
            }
            QFrame#StepBadge[state="pending"] {
                background: rgba(8, 22, 34, 0.82);
            }
            QFrame#StepBadge[state="current"] {
                background: rgba(19, 126, 165, 0.92);
                border-color: rgba(151, 240, 255, 0.42);
            }
            QFrame#StepBadge[state="complete"] {
                background: rgba(40, 118, 104, 0.88);
                border-color: rgba(147, 255, 216, 0.36);
            }
            QLabel#StepNumber {
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
                border-radius: 15px;
                background: rgba(244, 250, 255, 0.12);
                color: #f6fbff;
                font-size: 12px;
                font-weight: 700;
                qproperty-alignment: AlignCenter;
            }
            QLabel#StepTitle {
                font-size: 12px;
                font-weight: 700;
                color: #e4f6ff;
            }
            QLabel#PreviewFrame {
                background: rgba(2, 9, 15, 0.94);
                border: 1px solid rgba(24, 195, 255, 0.18);
                border-radius: 26px;
                color: rgba(234, 248, 255, 0.86);
                font-size: 14px;
                padding: 12px;
            }
            QScrollArea {
                background: transparent;
                border: 0;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.04);
                width: 10px;
                margin: 8px 0 8px 0;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(24, 195, 255, 0.34);
                border-radius: 5px;
                min-height: 28px;
            }
            """
        )

    def update_step_state(self, current_index: int) -> None:
        completed = {
            0: self.session_dir is not None,
            1: self.survey_completed,
            2: self.psl_report is not None,
            3: self._current_face_frame is not None or self.face_result is not None,
            4: self.face_result is not None,
            5: bool(self.psl_report or self.face_result or self.survey_result),
        }
        for index, badge in enumerate(self.step_badges):
            if index == current_index:
                state = "current"
            elif completed.get(index, False) or index < current_index:
                state = "complete"
            else:
                state = "pending"
            badge.set_state(state)

    def go_to_step(self, index: int) -> None:
        if index != 3:
            self.stop_face_preview()
        elif index == 3:
            self.refresh_face_cameras()
            self.start_face_preview()

        if index == 5:
            self.refresh_final_page()

        self.stacked.setCurrentIndexAnimated(index)
        self.update_step_state(index)

    def start_new_session(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "작업 진행 중", "아직 실행 중인 작업이 있습니다. 완료될 때까지 기다려주세요.")
            return

        self.reset_session_state()
        self.session_dir = new_health_rum_session_dir()
        self.main_session_label.setText(f"현재 세션: {self.session_dir.name}")
        self.go_to_step(1)

    def reset_session_state(self) -> None:
        self.stop_face_preview()
        self.survey_answers = {}
        self.survey_details = {}
        self.survey_result = None
        self.survey_completed = False
        self.psl_report = None
        self.psl_paths = {}
        self.face_result = None
        self.face_paths = {}
        self._current_face_frame = None

        self.main_session_label.setText("아직 생성된 세션이 없습니다.")
        self.psl_status_label.setText("PSL_Test 측정을 기다리는 중입니다.")
        self.face_capture_status_label.setText("실시간 얼굴 미리보기를 시작하기 전입니다.")
        self.face_review_status_label.setText("아직 Face_AI 결과가 없습니다.")
        self.final_status_label.setText("이 페이지를 채우려면 PSL_Test와 Face_AI 단계를 완료하세요.")

        self.psl_summary.clear()
        self.psl_log.clear()
        self.face_summary.clear()
        self.face_log.clear()
        self.final_summary_text.clear()
        self.final_paths_text.clear()
        if hasattr(self, "recommendation_text"):
            self.recommendation_text.setPlainText("체질 설문과 측정 결과가 쌓이면 이곳에 판정 해설이 표시됩니다.")
        if hasattr(self, "populate_final_report_placeholders"):
            self.populate_final_report_placeholders()

        self.face_preview_label.set_placeholder("여기에 실시간 얼굴 미리보기가 표시됩니다.")
        self.face_snapshot_label.set_placeholder("여기에 Face_AI 분석 스냅샷이 표시됩니다.")
        if hasattr(self, "survey_result_label"):
            self.survey_result_label.setText("기본 체질 미리보기: 문항을 하나 이상 선택하세요.")
        for group in SURVEY_GROUPS:
            for checkbox in self.survey_checkboxes.get(group["key"], []):
                was_blocked = checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(was_blocked)
            count_label = self.survey_count_labels.get(group["key"])
            if count_label is not None:
                count_label.setText(f"선택 항목: 0 / {len(group['items'])}")
        if hasattr(self, "survey_summary_text"):
            self.survey_summary_text.setPlainText(format_survey_summary({}, {}))

        self.psl_to_face_button.setEnabled(False)
        self.face_to_review_button.setEnabled(False)
        self.face_to_final_button.setEnabled(False)

        for card_group in (self.psl_cards, self.face_review_cards, self.final_cards):
            for card in card_group.values():
                card.reset()

    def refresh_ports(self) -> None:
        current = self.psl_port_combo.currentData() if hasattr(self, "psl_port_combo") else None
        ports = list_serial_ports()
        ports.sort(key=lambda item: (0 if "arduino" in item["description"].lower() else 1, item["device"]))
        self._port_entries = ports

        self.psl_port_combo.clear()
        if not ports:
            self.psl_port_combo.addItem("시리얼 포트를 찾지 못했습니다", None)
            return

        selected_index = 0
        for index, entry in enumerate(ports):
            label = f"{entry['device']} - {entry['description']}"
            self.psl_port_combo.addItem(label, entry["device"])
            if current and entry["device"] == current:
                selected_index = index
        self.psl_port_combo.setCurrentIndex(selected_index)

    def refresh_psl_cameras(self) -> None:
        self.refresh_all_cameras()
        return
        current = self.psl_camera_combo.currentData() if hasattr(self, "psl_camera_combo") else None
        cameras = probe_camera_indices(max_index=4)
        self.psl_camera_combo.clear()
        if not cameras:
            self.psl_camera_combo.addItem("카메라를 찾지 못했습니다", None)
            return

        selected_index = 0
        for index, camera in enumerate(cameras):
            label = f"카메라 {camera['index']} | {camera['width']}x{camera['height']} | {safe_float(camera['fps']):.1f} fps"
            self.psl_camera_combo.addItem(label, int(camera["index"]))
            if current is not None and int(camera["index"]) == int(current):
                selected_index = index
        self.psl_camera_combo.setCurrentIndex(selected_index)

    def refresh_face_cameras(self) -> None:
        self.refresh_all_cameras()
        return
        current = self.face_camera_combo.currentData() if hasattr(self, "face_camera_combo") else None
        preferred = self.psl_camera_combo.currentData() if hasattr(self, "psl_camera_combo") else None
        cameras = probe_camera_indices(max_index=4)
        self.face_camera_combo.clear()
        if not cameras:
            self.face_camera_combo.addItem("카메라를 찾지 못했습니다", None)
            return

        selected_index = 0
        for index, camera in enumerate(cameras):
            label = f"카메라 {camera['index']} | {camera['width']}x{camera['height']} | {safe_float(camera['fps']):.1f} fps"
            camera_value = int(camera["index"])
            self.face_camera_combo.addItem(label, camera_value)
            if current is not None and camera_value == int(current):
                selected_index = index
            elif current is None and preferred is not None and camera_value == int(preferred):
                selected_index = index
        self.face_camera_combo.setCurrentIndex(selected_index)

    def _populate_camera_combo(self, combo: QComboBox, cameras: list[dict[str, int | float]], current, preferred=None) -> None:
        combo.clear()
        if not cameras:
            combo.addItem("카메라를 찾지 못했습니다", None)
            return

        selected_index = 0
        for index, camera in enumerate(cameras):
            label = f"카메라 {camera['index']} | {camera['width']}x{camera['height']} | {safe_float(camera['fps']):.1f} fps"
            camera_value = int(camera["index"])
            combo.addItem(label, camera_value)
            if current is not None and camera_value == int(current):
                selected_index = index
            elif current is None and preferred is not None and camera_value == int(preferred):
                selected_index = index
        combo.setCurrentIndex(selected_index)

    def refresh_all_cameras(self) -> None:
        current_psl = self.psl_camera_combo.currentData() if hasattr(self, "psl_camera_combo") else None
        current_face = self.face_camera_combo.currentData() if hasattr(self, "face_camera_combo") else None
        preferred_face = current_psl if current_psl is not None else current_face

        cameras = probe_camera_indices(max_index=4)
        self._camera_entries = cameras
        if hasattr(self, "psl_camera_combo"):
            self._populate_camera_combo(self.psl_camera_combo, cameras, current_psl)
        if hasattr(self, "face_camera_combo"):
            self._populate_camera_combo(self.face_camera_combo, cameras, current_face, preferred_face)

    def collect_survey_answers(self) -> dict[str, int]:
        answers: dict[str, int] = {}
        for group in SURVEY_GROUPS:
            selected_items = self.collect_survey_details_for_group(group["key"])
            answers[group["key"]] = len(selected_items)
        return answers

    def collect_survey_details_for_group(self, group_key: str) -> list[str]:
        selected_items: list[str] = []
        for checkbox in self.survey_checkboxes.get(group_key, []):
            if checkbox.isChecked():
                selected_items.append(checkbox.text())
        return selected_items

    def collect_survey_details(self) -> dict[str, list[str]]:
        return {
            group["key"]: self.collect_survey_details_for_group(group["key"])
            for group in SURVEY_GROUPS
        }

    def update_survey_preview(self) -> None:
        self.survey_answers = self.collect_survey_answers()
        self.survey_details = self.collect_survey_details()

        for group in SURVEY_GROUPS:
            count_label = self.survey_count_labels.get(group["key"])
            if count_label is not None:
                count_label.setText(f"선택 항목: {self.survey_answers.get(group['key'], 0)} / {len(group['items'])}")

        survey_summary = format_survey_summary(self.survey_answers, self.survey_details)
        if hasattr(self, "survey_summary_text"):
            self.survey_summary_text.setPlainText(survey_summary)

        total_selected = sum(self.survey_answers.values())
        if total_selected <= 0:
            self.survey_result = None
            if hasattr(self, "survey_result_label"):
                self.survey_result_label.setText("기본 체질 미리보기: 문항을 하나 이상 선택하세요.")
            return

        preview_result = build_profile_recommendation(self.survey_answers, self.psl_report, self.face_result, self.survey_details)
        self.survey_result = preview_result
        if hasattr(self, "survey_result_label"):
            self.survey_result_label.setText(
                f"기본 체질 미리보기: {preview_result['constitution_label']} | 예상 타입: {preview_result['profile_label']}"
            )

    def complete_survey_and_continue(self) -> None:
        self.update_survey_preview()
        if sum(self.survey_answers.values()) <= 0:
            QMessageBox.warning(self, "설문 선택 필요", "체질 설문 문항을 하나 이상 선택한 뒤 다음 단계로 진행하세요.")
            return
        self.survey_completed = True
        self.go_to_step(2)

    def selected_port(self) -> str | None:
        port = self.psl_port_combo.currentData()
        return str(port) if port else None

    def build_psl_config(self) -> HealthRumPslConfig:
        age_value = int(self.psl_age_spin.value())
        calibration_sbp = parse_optional_float(self.psl_sbp_input.text())
        calibration_dbp = parse_optional_float(self.psl_dbp_input.text())
        camera_index = self.psl_camera_combo.currentData()
        return HealthRumPslConfig(
            mode=str(self.psl_mode_combo.currentData() or "ppg"),
            port=self.selected_port(),
            duration_s=float(self.psl_duration_spin.value()),
            age=age_value if age_value > 0 else None,
            sex=str(self.psl_sex_combo.currentData() or "unknown"),
            calibration_sbp=calibration_sbp,
            calibration_dbp=calibration_dbp,
            camera_index=int(camera_index) if camera_index is not None else None,
        )

    def start_psl_measurement(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "작업 진행 중", "이미 다른 작업이 실행 중입니다.")
            return

        if self.session_dir is None:
            self.session_dir = new_health_rum_session_dir()
            self.main_session_label.setText(f"현재 세션: {self.session_dir.name}")

        try:
            config = self.build_psl_config()
        except ValueError as exc:
            QMessageBox.warning(self, "입력 오류", str(exc))
            return

        if not config.port:
            QMessageBox.warning(self, "포트 선택 필요", "PSL_Test를 시작하기 전에 Arduino 시리얼 포트를 선택하세요.")
            return
        if config.mode == "multimodal" and config.camera_index is None:
            QMessageBox.warning(self, "카메라 선택 필요", "카메라 + PPG 모드를 사용하려면 카메라를 선택하세요.")
            return

        self.psl_log.clear()
        self.psl_summary.clear()
        self.psl_status_label.setText("PSL_Test 실행 중...")
        self.psl_start_button.setEnabled(False)
        self.psl_to_face_button.setEnabled(False)

        thread = QThread(self)
        worker = HealthRumPslWorker(config, self.session_dir)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.append_psl_log)
        worker.completed.connect(self.on_psl_completed)
        worker.failed.connect(self.on_psl_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.cleanup_worker)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        thread.start()

    def append_psl_log(self, message: str) -> None:
        self.psl_status_label.setText(message)
        self.psl_log.appendPlainText(message)

    def on_psl_completed(self, report: dict[str, Any], paths: dict[str, str]) -> None:
        self.psl_report = report
        self.psl_paths = dict(paths)
        self.update_survey_preview()
        self.update_psl_cards()
        self.psl_summary.setPlainText(summarize_psl_report(report))
        self.psl_status_label.setText("PSL_Test가 완료되었습니다. 얼굴 촬영 단계로 이동할 수 있습니다.")
        self.psl_start_button.setEnabled(True)
        self.psl_to_face_button.setEnabled(True)
        self.go_to_step(3)

    def on_psl_failed(self, error_text: str) -> None:
        self.psl_status_label.setText("PSL_Test가 실패했습니다.")
        self.psl_start_button.setEnabled(True)
        self.psl_to_face_button.setEnabled(False)
        self.psl_log.appendPlainText(f"오류: {error_text}")
        QMessageBox.critical(self, "PSL_Test 실패", error_text)

    def update_psl_cards(self) -> None:
        if not self.psl_report:
            for card in self.psl_cards.values():
                card.reset()
            return

        heart = self.psl_report.get("heart_rate") or {}
        hrv = self.psl_report.get("hrv") or {}
        stress = self.psl_report.get("stress") or {}
        bp = self.psl_report.get("blood_pressure") or {}

        self.psl_cards["heart_rate"].update_card(
            f"{safe_float(heart.get('heart_rate_bpm')):.1f} bpm",
            f"IBI {safe_float(heart.get('ibi_mean_ms')):.0f} ms",
        )
        self.psl_cards["hrv"].update_card(
            f"RMSSD {safe_float(hrv.get('rmssd_ms')):.1f}",
            f"SDNN {safe_float(hrv.get('sdnn_ms')):.1f} ms",
        )
        self.psl_cards["stress"].update_card(
            f"{safe_float(stress.get('stress_score')):.1f}",
            str(stress.get("stress_state") or "상태 없음"),
        )
        self.psl_cards["blood_pressure"].update_card(
            f"{safe_float(bp.get('estimated_sbp')):.0f}/{safe_float(bp.get('estimated_dbp')):.0f}",
                str(bp.get("blood_pressure_trend") or "추세 없음"),
        )

    def current_face_camera_index(self) -> int | None:
        camera_index = self.face_camera_combo.currentData()
        return int(camera_index) if camera_index is not None else None

    def start_face_preview(self) -> None:
        camera_index = self.current_face_camera_index()
        if camera_index is None:
            self.face_preview_label.set_placeholder("Face_AI용 카메라가 선택되지 않았습니다.")
            return

        if self._face_capture is not None and self._face_capture.isOpened():
            return

        capture = open_camera_capture(camera_index, width=1280, height=720, fps=30.0)
        if capture is None:
            self.face_preview_label.set_placeholder(f"카메라 {camera_index}를 열 수 없습니다.")
            return

        self._face_capture = capture
        self._face_timer.start()
        self.face_capture_status_label.setText(f"카메라 {camera_index}에서 실시간 미리보기를 시작했습니다.")
        self.face_log.appendPlainText(f"카메라 {camera_index}에서 미리보기를 시작했습니다.")

    def stop_face_preview(self) -> None:
        self._face_timer.stop()
        if self._face_capture is not None:
            self._face_capture.release()
            self._face_capture = None

    def restart_face_preview(self) -> None:
        self.stop_face_preview()
        self.start_face_preview()

    def update_face_preview(self) -> None:
        if self._face_capture is None:
            return

        ok, frame = self._face_capture.read()
        if not ok or frame is None:
            self.face_preview_label.set_placeholder("카메라 프레임을 가져올 수 없습니다.")
            return

        self._current_face_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QImage(rgb_frame.data, width, height, channels * width, image_format_rgb888()).copy()
        self.face_preview_label.set_preview_pixmap(QPixmap.fromImage(image))

    def start_face_analysis(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "작업 진행 중", "이미 다른 작업이 실행 중입니다.")
            return
        if self.session_dir is None:
            QMessageBox.warning(self, "세션 필요", "Face_AI를 실행하기 전에 먼저 세션을 시작하세요.")
            return
        if self._current_face_frame is None:
            QMessageBox.warning(self, "프레임 없음", "먼저 미리보기를 시작하고 얼굴을 중앙에 맞춰주세요.")
            return

        self.face_log.clear()
        self.face_capture_status_label.setText("현재 얼굴 프레임을 분석하는 중입니다...")
        self.face_review_status_label.setText("Face_AI가 촬영된 프레임을 분석하는 중입니다...")
        self.face_analyze_button.setEnabled(False)
        self.face_to_review_button.setEnabled(False)
        self.face_to_final_button.setEnabled(False)

        thread = QThread(self)
        worker = FaceAiAnalyzeWorker(self._current_face_frame.copy(), self.session_dir)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.append_face_log)
        worker.completed.connect(self.on_face_completed)
        worker.failed.connect(self.on_face_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.cleanup_worker)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        thread.start()

    def append_face_log(self, message: str) -> None:
        self.face_capture_status_label.setText(message)
        self.face_log.appendPlainText(message)

    def on_face_completed(self, result: dict[str, Any], paths: dict[str, str]) -> None:
        self.face_result = result
        self.face_paths = dict(paths)
        self.update_survey_preview()
        self.face_analyze_button.setEnabled(True)
        self.face_to_review_button.setEnabled(True)
        self.face_to_final_button.setEnabled(True)

        self.update_face_review_page()
        if result.get("face_detected"):
            self.face_capture_status_label.setText("Face_AI 분석이 완료되었습니다. 다음 페이지에서 확인하세요.")
        else:
            message = str(result.get("message") or "얼굴을 감지하지 못했습니다.")
            self.face_capture_status_label.setText(f"Face_AI 분석은 끝났지만 얼굴이 감지되지 않았습니다. {message}")
        self.go_to_step(4)

    def on_face_failed(self, error_text: str) -> None:
        self.face_analyze_button.setEnabled(True)
        self.face_to_review_button.setEnabled(False)
        self.face_to_final_button.setEnabled(False)
        self.face_log.appendPlainText(f"오류: {error_text}")
        self.face_capture_status_label.setText("Face_AI 분석에 실패했습니다.")
        self.face_review_status_label.setText("Face_AI 분석에 실패했습니다.")
        QMessageBox.critical(self, "Face_AI 실패", error_text)

    def update_face_review_page(self) -> None:
        if not self.face_result:
            for card in self.face_review_cards.values():
                card.reset()
            self.face_summary.setPlainText("아직 Face_AI 결과가 없습니다.")
            self.face_snapshot_label.set_placeholder("여기에 Face_AI 분석 스냅샷이 표시됩니다.")
            self.face_review_status_label.setText("아직 Face_AI 결과가 없습니다.")
            return

        if not self.face_result.get("face_detected", True):
            message = str(self.face_result.get("message") or "얼굴을 감지하지 못했습니다.")
            self.face_review_cards["overall"].update_card("미감지", message)
            for key in ("wrinkle", "pigmentation", "pore", "dryness", "sagging"):
                self.face_review_cards[key].update_card("측정 불가", "얼굴 미감지")
            self.face_summary.setPlainText(summarize_face_result(self.face_result))
            self.face_review_status_label.setText(f"얼굴 미감지 | {message}")

            snapshot_path = self.face_paths.get("snapshot_path")
            if snapshot_path and Path(snapshot_path).exists():
                pixmap = QPixmap(snapshot_path)
                if not pixmap.isNull():
                    self.face_snapshot_label.set_preview_pixmap(pixmap)
                    return
            self.face_snapshot_label.set_placeholder("여기에 Face_AI 분석 스냅샷이 표시됩니다.")
            return

        metrics = self.face_result.get("metrics") or {}
        overall_score = self.face_result.get("overall_score")
        overall_label = self.face_result.get("overall_label") or "상태 없음"
        self.face_review_cards["overall"].update_card(
            f"{safe_float(overall_score):.1f}" if overall_score is not None else "-",
            overall_label,
        )

        for key in ("wrinkle", "pigmentation", "pore", "dryness", "sagging"):
            metric = metrics.get(key)
            if not metric:
                self.face_review_cards[key].reset()
                continue
            self.face_review_cards[key].update_card(
                f"{safe_float(metric.get('score')):.1f}",
                str(metric.get("severity_label") or "상태 없음"),
            )

        self.face_summary.setPlainText(summarize_face_result(self.face_result))
        detected = "얼굴 감지됨" if self.face_result.get("face_detected", True) else "얼굴 미감지"
        self.face_review_status_label.setText(
            f"{detected} | 분석 모드: {self.face_result.get('calibration_mode') or '-'}"
        )

        snapshot_path = self.face_paths.get("snapshot_path")
        if snapshot_path and Path(snapshot_path).exists():
            pixmap = QPixmap(snapshot_path)
            if not pixmap.isNull():
                self.face_snapshot_label.set_preview_pixmap(pixmap)
                return
        self.face_snapshot_label.set_placeholder("분석 스냅샷 파일을 찾지 못했습니다.")

    def refresh_final_page(self) -> None:
        if sum(self.survey_answers.values()) > 0:
            self.survey_result = build_profile_recommendation(self.survey_answers, self.psl_report, self.face_result, self.survey_details)
            self.recommendation_text.setPlainText(format_profile_recommendation(self.survey_result))
        else:
            self.survey_result = None
            self.recommendation_text.setPlainText("체질 설문을 완료하면 이 영역에 상세 판정 해설이 표시됩니다.")

        self.final_biosignal_table.set_table(
            ["측정 항목", "측정 데이터", "정상 범위", "AI 분석 결과", "고객에게 설명 자료"],
            self.build_biosignal_report_rows(),
            column_stretches=[16, 16, 13, 18, 31],
        )
        self.final_skin_table.set_table(
            ["측정 항목", "측정 데이터", "AI 분석 지수", "피부 상태 평가", "추천 관리 방향"],
            self.build_skin_report_rows(),
            column_stretches=[16, 23, 16, 18, 27],
        )
        zang_rows, fu_rows = self.build_organ_balance_rows()
        self.final_zang_table.set_table(
            ["5 장부 영역", "AI 밸런스 점수", "상태 평가", "의미", "생활 관리 제안"],
            zang_rows,
            caption="5장부 영역",
            column_stretches=[15, 18, 16, 22, 29],
        )
        self.final_fu_table.set_table(
            ["6 부 영역", "AI 밸런스 점수", "상태 평가", "의미", "생활 관리 제안"],
            fu_rows,
            caption="6부 영역",
            column_stretches=[15, 18, 16, 22, 29],
        )
        constitution_rows, emphasized_rows = self.build_constitution_report_rows()
        self.final_constitution_table.set_table(
            ["기본 4성 체질", "8성 체질", "고객의 체질 특징 설명 (8개에서 1가지)", "체질 판별"],
            constitution_rows,
            column_stretches=[14, 22, 42, 18],
            emphasized_rows=emphasized_rows,
        )

        if self.psl_report and self.face_result and not self.face_result.get("face_detected", True):
            self.final_status_label.setText("PSL_Test는 완료되었지만 Face_AI에서 얼굴을 감지하지 못했습니다. 얼굴을 중앙에 맞춰 다시 촬영하면 피부 리포트가 채워집니다.")
        elif self.psl_report and self.face_result:
            self.final_status_label.setText("설문, PSL_Test, Face_AI를 모두 반영한 표 형식 통합 리포트가 준비되었습니다.")
        elif self.psl_report:
            self.final_status_label.setText("심혈관 리포트는 준비되었고, 피부 리포트는 Face_AI 분석을 실행하면 채워집니다.")
        elif self.face_result:
            self.final_status_label.setText("피부 리포트는 준비되었고, 심혈관 리포트는 PSL_Test를 실행하면 채워집니다.")
        else:
            self.final_status_label.setText("설문, PSL_Test, Face_AI를 모두 완료하면 최종 보고서형 결과가 채워집니다.")

        self.final_summary_text.setPlainText(self.build_integrated_summary())

        path_lines = []
        if self.session_dir is not None:
            path_lines.append(f"세션 폴더: {self.session_dir}")
        for prefix, payload in (
            ("survey", {"summary": format_survey_summary(self.survey_answers, self.survey_details)}),
            ("psl", self.psl_paths),
            ("face", self.face_paths),
        ):
            for key, value in payload.items():
                prefix_label = "설문" if prefix == "survey" else "PSL" if prefix == "psl" else "Face"
                path_lines.append(f"{prefix_label}.{key}: {value}")
        self.final_paths_text.setPlainText("\n".join(path_lines))
        self.animate_final_report_sections()

    def build_integrated_summary(self) -> str:
        recommendation = self.survey_result
        if recommendation is None and sum(self.survey_answers.values()) > 0:
            recommendation = build_profile_recommendation(
                self.survey_answers,
                self.psl_report,
                self.face_result,
                self.survey_details,
            )
        lines = ["헬스럼 통합 요약", "================"]
        if self.session_dir is not None:
            lines.append(f"세션: {self.session_dir.name}")

        lines.extend(["", "[체질 설문]", format_survey_summary(self.survey_answers, self.survey_details)])
        lines.extend(["", "[체질 및 기기 추천]", format_profile_recommendation(recommendation)])
        lines.extend(["", "[PSL_Test]", summarize_psl_report(self.psl_report)])
        lines.extend(["", "[Face_AI]", summarize_face_result(self.face_result)])
        return "\n".join(lines)

    def export_combined_summary(self) -> None:
        if self.session_dir is None:
            QMessageBox.warning(self, "세션 없음", "먼저 세션을 시작하세요.")
            return

        text_path = self.session_dir / "health_rum_summary.txt"
        json_path = self.session_dir / "health_rum_summary.json"

        summary_payload = {
            "session_dir": str(self.session_dir),
            "survey_answers": self.survey_answers,
            "survey_details": self.survey_details,
            "survey_result": self.survey_result
            or (
                build_profile_recommendation(
                    self.survey_answers,
                    self.psl_report,
                    self.face_result,
                    self.survey_details,
                )
                if sum(self.survey_answers.values()) > 0
                else None
            ),
            "psl_report": self.psl_report,
            "psl_paths": self.psl_paths,
            "face_result": self.face_result,
            "face_paths": self.face_paths,
        }

        text_path.write_text(self.build_integrated_summary(), encoding="utf-8")
        write_json(json_path, summary_payload)
        self.refresh_final_page()
        QMessageBox.information(self, "저장 완료", f"저장 위치:\n{text_path}\n{json_path}")

    def cleanup_worker(self) -> None:
        self._worker = None
        self._thread = None
        self.psl_start_button.setEnabled(True)
        self.face_analyze_button.setEnabled(True)

    def open_session_dir(self) -> None:
        if self.session_dir is None:
            QMessageBox.warning(self, "세션 없음", "아직 생성된 세션 폴더가 없습니다.")
            return
        try:
            os.startfile(str(self.session_dir))
        except AttributeError:
            QMessageBox.information(self, "세션 폴더", str(self.session_dir))

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_face_preview()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(1500)
        super().closeEvent(event)


def main() -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)

    window = HealthRumWindow()
    window.show()

    if not owns_app:
        return 0
    exec_fn = getattr(app, "exec", None)
    if exec_fn is None:
        exec_fn = app.exec_
    return exec_fn()
