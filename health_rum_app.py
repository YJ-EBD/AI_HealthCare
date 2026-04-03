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
    KEEP_ASPECT = Qt.AspectRatioMode.KeepAspectRatio
    SMOOTH_TRANSFORM = Qt.TransformationMode.SmoothTransformation
    SCROLLBAR_OFF = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
except AttributeError:
    ALIGN_CENTER = Qt.AlignCenter
    ALIGN_TOP = Qt.AlignTop
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
        layout.setSpacing(12)

        final_panel = QFrame()
        final_panel.setObjectName("Panel")
        final_layout = QVBoxLayout(final_panel)
        final_layout.setContentsMargins(18, 18, 18, 18)
        final_layout.setSpacing(12)

        title = QLabel("6. 최종 통합 결과")
        title.setObjectName("SectionTitle")
        desc = QLabel(
            "이 페이지에서만 PSL_Test의 모든 값과 Face_AI의 모든 값이 한 번에 함께 표시됩니다."
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

        recommendation_group = QGroupBox("체질 및 기기 추천")
        recommendation_layout = QGridLayout(recommendation_group)
        recommendation_meta = {
            "constitution": ("기본 체질", "#9ddfff"),
            "profile_type": ("추천 타입", "#49e1b6"),
        }
        for idx, (key, meta) in enumerate(recommendation_meta.items()):
            card = DashboardCard(meta[0], meta[1])
            self.final_cards[key] = card
            recommendation_layout.addWidget(card, 0, idx)
        self.recommendation_text = QPlainTextEdit()
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setMinimumHeight(260)
        recommendation_layout.addWidget(self.recommendation_text, 1, 0, 1, 2)
        layout.addWidget(recommendation_group)

        biosignal_group = QGroupBox("PSL_Test 결과")
        biosignal_layout = QGridLayout(biosignal_group)
        psl_final_cards = {
            "heart_rate": ("심박수", "#18c3ff"),
            "hrv": ("HRV", "#ff6b6b"),
            "stress": ("스트레스", "#f7a23b"),
            "circulation": ("혈류 순환", "#49e1b6"),
            "vascular_health": ("혈관 건강", "#88f1ff"),
            "vascular_age": ("혈관 나이", "#7cb8ff"),
            "blood_pressure": ("혈압 추정", "#ffe082"),
        }
        for idx, (key, meta) in enumerate(psl_final_cards.items()):
            card = DashboardCard(meta[0], meta[1])
            self.final_cards[key] = card
            biosignal_layout.addWidget(card, idx // 2, idx % 2)
        layout.addWidget(biosignal_group)

        face_group = QGroupBox("Face_AI 결과")
        face_layout = QGridLayout(face_group)
        face_final_cards = {
            "face_overall": ("얼굴 종합", "#49e1b6"),
            "face_top": ("주요 항목", "#ff7d5b"),
            "wrinkle": ("주름", "#ff7d5b"),
            "pigmentation": ("색소", "#f7c65a"),
            "pore": ("모공", "#49b6ff"),
            "dryness": ("건조", "#ff9db3"),
            "sagging": ("처짐", "#a2f56f"),
        }
        for idx, (key, meta) in enumerate(face_final_cards.items()):
            card = DashboardCard(meta[0], meta[1])
            self.final_cards[key] = card
            face_layout.addWidget(card, idx // 2, idx % 2)
        layout.addWidget(face_group)

        self.final_summary_text = QPlainTextEdit()
        self.final_summary_text.setReadOnly(True)
        self.final_summary_text.setMinimumHeight(320)
        self.final_paths_text = QPlainTextEdit()
        self.final_paths_text.setReadOnly(True)
        self.final_paths_text.setMinimumHeight(180)
        layout.addWidget(self.wrap_group("통합 세션 요약", self.final_summary_text))
        layout.addWidget(self.wrap_group("저장된 파일", self.final_paths_text))
        return page

    def wrap_group(self, title: str, widget: QWidget) -> QGroupBox:
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(12, 12, 12, 12)
        group_layout.addWidget(widget)
        return group

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
            self.recommendation_text.clear()

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
        self.survey_result = build_profile_recommendation(self.survey_answers, self.psl_report, self.face_result, self.survey_details)
        self.final_cards["constitution"].update_card(
            self.survey_result["constitution_label"],
            "설문 + 바이오신호 + Face_AI 종합",
        )
        self.final_cards["profile_type"].update_card(
            self.survey_result["profile_label"],
            self.survey_result["goal"],
        )
        self.recommendation_text.setPlainText(format_profile_recommendation(self.survey_result))

        if self.psl_report:
            heart = self.psl_report.get("heart_rate") or {}
            hrv = self.psl_report.get("hrv") or {}
            stress = self.psl_report.get("stress") or {}
            circulation = self.psl_report.get("circulation") or {}
            vascular_health = self.psl_report.get("vascular_health") or {}
            vascular_age = self.psl_report.get("vascular_age") or {}
            bp = self.psl_report.get("blood_pressure") or {}

            self.final_cards["heart_rate"].update_card(
                f"{safe_float(heart.get('heart_rate_bpm')):.1f} bpm",
                f"신호 품질 {safe_float((self.psl_report.get('metadata') or {}).get('signal_quality_score')):.1f}",
            )
            self.final_cards["hrv"].update_card(
                f"RMSSD {safe_float(hrv.get('rmssd_ms')):.1f}",
                f"SDNN {safe_float(hrv.get('sdnn_ms')):.1f} ms",
            )
            self.final_cards["stress"].update_card(
                f"{safe_float(stress.get('stress_score')):.1f}",
                str(stress.get("stress_state") or "상태 없음"),
            )
            self.final_cards["circulation"].update_card(
                f"{safe_float(circulation.get('circulation_score')):.1f}",
                f"관류지수 {safe_float(circulation.get('perfusion_index')):.3f}",
            )
            self.final_cards["vascular_health"].update_card(
                f"{safe_float(vascular_health.get('vascular_health_score')):.1f}",
                f"반사 지수 {safe_float(vascular_health.get('reflection_index')):.2f}",
            )
            self.final_cards["vascular_age"].update_card(
                f"{safe_float(vascular_age.get('vascular_age_estimate')):.1f}",
                f"차이 {safe_float(vascular_age.get('vascular_age_gap')):+.1f}",
            )
            self.final_cards["blood_pressure"].update_card(
                f"{safe_float(bp.get('estimated_sbp')):.0f}/{safe_float(bp.get('estimated_dbp')):.0f}",
                str(bp.get("blood_pressure_trend") or "추세 없음"),
            )
        else:
            for key in (
                "heart_rate",
                "hrv",
                "stress",
                "circulation",
                "vascular_health",
                "vascular_age",
                "blood_pressure",
            ):
                self.final_cards[key].reset()

        if self.face_result:
            if not self.face_result.get("face_detected", True):
                message = str(self.face_result.get("message") or "얼굴을 감지하지 못했습니다.")
                self.final_cards["face_overall"].update_card("미감지", message)
                self.final_cards["face_top"].update_card("재시도", "얼굴을 중앙에 맞춰 다시 촬영하세요")
                for key in ("wrinkle", "pigmentation", "pore", "dryness", "sagging"):
                    self.final_cards[key].update_card("측정 불가", "얼굴 미감지")
            else:
                metrics = self.face_result.get("metrics") or {}
                overall_score = self.face_result.get("overall_score")
                overall_label = self.face_result.get("overall_label") or "상태 없음"
                top_score, top_title = top_face_concern(self.face_result)

                self.final_cards["face_overall"].update_card(
                    f"{safe_float(overall_score):.1f}" if overall_score is not None else "-",
                    overall_label,
                )
                self.final_cards["face_top"].update_card(top_score, top_title)
                for key in ("wrinkle", "pigmentation", "pore", "dryness", "sagging"):
                    metric = metrics.get(key)
                    if not metric:
                        self.final_cards[key].reset()
                        continue
                    self.final_cards[key].update_card(
                        f"{safe_float(metric.get('score')):.1f}",
                        str(metric.get("severity_label") or "상태 없음"),
                    )
        else:
            for key in ("face_overall", "face_top", "wrinkle", "pigmentation", "pore", "dryness", "sagging"):
                self.final_cards[key].reset()

        if self.psl_report and self.face_result and not self.face_result.get("face_detected", True):
            self.final_status_label.setText("PSL_Test는 완료되었지만 Face_AI에서 얼굴을 감지하지 못했습니다. 얼굴을 중앙에 맞춰 다시 촬영해보세요.")
        elif self.psl_report and self.face_result:
            self.final_status_label.setText("설문, PSL_Test, Face_AI를 모두 반영한 최종 결과가 준비되었습니다.")
        elif self.psl_report:
            self.final_status_label.setText("설문과 PSL_Test는 반영되었습니다. 얼굴 영역 결과를 채우려면 Face_AI를 실행하세요.")
        elif self.face_result:
            self.final_status_label.setText("설문과 Face_AI는 반영되었습니다. 바이오신호 영역 결과를 채우려면 PSL_Test를 실행하세요.")
        else:
            self.final_status_label.setText("설문, PSL_Test, Face_AI를 모두 완료하면 최종 판정이 채워집니다.")

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

    def build_integrated_summary(self) -> str:
        recommendation = self.survey_result or build_profile_recommendation(
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
            "survey_result": self.survey_result or build_profile_recommendation(
                self.survey_answers,
                self.psl_report,
                self.face_result,
                self.survey_details,
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
