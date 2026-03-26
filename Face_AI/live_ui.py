from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from live_runtime import (
    METRIC_META,
    TASK_META,
    LiveSkinAnalyzer,
    build_default_paths,
    draw_analysis_overlay,
    save_image_unicode,
)


class ProgressBar(QFrame):
    def __init__(self, accent: str) -> None:
        super().__init__()
        self.setObjectName("ProgressBarShell")
        self.setMinimumHeight(10)
        self.setMaximumHeight(10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fill = QFrame()
        self.fill.setObjectName("ProgressBarFill")
        self.fill.setMinimumHeight(10)
        self.fill.setMaximumHeight(10)
        self.fill.setStyleSheet(
            f"background-color: {accent}; border-radius: 5px;"
        )
        layout.addWidget(self.fill)
        self._percent = 0
        self.update_fill(0)

    def update_fill(self, percent: float) -> None:
        self._percent = max(0.0, min(100.0, float(percent)))
        parent_width = max(1, self.width())
        fill_width = int(parent_width * (self._percent / 100.0))
        self.fill.setFixedWidth(fill_width)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_fill(self._percent)


class SummaryCard(QFrame):
    def __init__(self, title: str, accent: str) -> None:
        super().__init__()
        self.accent = accent
        self.setObjectName("SummaryCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("CardTitle")

        self.score_label = QLabel("대기 중")
        self.score_label.setObjectName("CardValue")

        self.detail_label = QLabel("카메라를 시작하면 분석이 표시됩니다.")
        self.detail_label.setObjectName("CardDetail")
        self.detail_label.setWordWrap(True)

        self.bar = ProgressBar(accent)

        layout.addWidget(self.title_label)
        layout.addWidget(self.score_label)
        layout.addWidget(self.bar)
        layout.addWidget(self.detail_label)

    def reset(self) -> None:
        self.score_label.setText("대기 중")
        self.detail_label.setText("카메라를 시작하면 분석이 표시됩니다.")
        self.bar.update_fill(0)

    def update_metric(self, payload: dict | None) -> None:
        if not payload:
            self.reset()
            return

        score = payload["score"]
        self.score_label.setText(f"{score:.1f}")
        self.detail_label.setText(f"{payload['severity_label']} 상대 지수")
        self.bar.update_fill(score)


class TaskCard(QFrame):
    def __init__(self, title: str, accent: str) -> None:
        super().__init__()
        self.accent = accent
        self.setObjectName("TaskCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("CardTitle")

        self.level_label = QLabel("분석 대기")
        self.level_label.setObjectName("TaskLevel")

        self.bar = ProgressBar(accent)

        self.detail_label = QLabel("얼굴을 중앙에 맞추면 자동으로 갱신됩니다.")
        self.detail_label.setObjectName("CardDetail")
        self.detail_label.setWordWrap(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.level_label)
        layout.addWidget(self.bar)
        layout.addWidget(self.detail_label)

    def reset(self) -> None:
        self.level_label.setText("분석 대기")
        self.detail_label.setText("얼굴을 중앙에 맞추면 자동으로 갱신됩니다.")
        self.bar.update_fill(0)

    def update_task(self, payload: dict | None) -> None:
        if not payload:
            self.reset()
            return

        self.level_label.setText(
            f"{payload['severity_label']}  |  등급 {payload['pred_level']}/{payload['class_count']}"
        )
        if payload.get("source") == "reference_prior":
            self.detail_label.setText(
                f"기준 분포 보정 · 상대 지수 {payload['normalized_score']:.1f} · 모델 신뢰도 {payload['confidence']:.1f}%"
            )
        else:
            self.detail_label.setText(
                f"신뢰도 {payload['confidence']:.1f}% · 상대 지수 {payload['normalized_score']:.1f}"
            )
        self.bar.update_fill(payload["normalized_score"])


class InferenceThread(QThread):
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, analyzer: LiveSkinAnalyzer, frame_bgr) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.frame_bgr = frame_bgr

    def run(self) -> None:
        try:
            result = self.analyzer.analyze_frame(self.frame_bgr)
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.failed.emit(str(exc))
            return

        self.result_ready.emit(result)


class FaceAILiveWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        defaults = build_default_paths()
        self.snapshot_dir = Path(args.snapshot_dir or defaults["snapshot_dir"]).resolve()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.capture = None
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_frame)

        self.analyzer = LiveSkinAnalyzer(
            checkpoint_root=args.checkpoint_root,
            use_reference_calibration=not args.disable_reference_calibration,
        )
        self.inference_thread = None
        self.inference_busy = False
        self.last_inference_time = 0.0
        self.latest_frame = None
        self.latest_result = None

        self.metric_cards = {}
        self.task_cards = {}

        self.build_window()
        self.build_ui()
        self.populate_camera_selector()
        self.start_selected_camera()

    def build_window(self) -> None:
        self.setWindowTitle("Face AI Live Skin Studio")
        self.resize(self.args.window_width, self.args.window_height)
        self.setMinimumSize(720, 1080)
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f4efe7;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QWidget#Canvas {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #f6f0e7,
                    stop: 0.45 #efe2d2,
                    stop: 1 #eadac6
                );
            }
            QFrame#HeroPanel {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #18222b,
                    stop: 0.5 #243847,
                    stop: 1 #3f564a
                );
                border-radius: 30px;
            }
            QFrame#PreviewPanel {
                background: #11161c;
                border-radius: 30px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            QFrame#ControlPanel {
                background: rgba(255, 251, 245, 0.82);
                border-radius: 24px;
                border: 1px solid #d6c5b0;
            }
            QFrame#SummaryCard, QFrame#TaskCard {
                background: rgba(255, 252, 247, 0.96);
                border-radius: 24px;
                border: 1px solid #e2d3bf;
            }
            QFrame#ProgressBarShell {
                background: #eadfce;
                border-radius: 5px;
            }
            QLabel#HeroTitle {
                color: #fff7ea;
                font-size: 34px;
                font-weight: 700;
            }
            QLabel#HeroSubTitle {
                color: #dae7ef;
                font-size: 15px;
            }
            QLabel#StatusPill {
                color: #18222b;
                background: #f4d8a1;
                border-radius: 16px;
                padding: 8px 14px;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#SectionTitle {
                color: #18222b;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#SectionSubTitle {
                color: #4b5d66;
                font-size: 13px;
            }
            QLabel#PreviewLabel {
                color: #f5eee0;
                background: #11161c;
                border-radius: 24px;
            }
            QLabel#CardTitle {
                color: #2c3c45;
                font-size: 13px;
                font-weight: 700;
            }
            QLabel#CardValue {
                color: #18222b;
                font-size: 30px;
                font-weight: 700;
            }
            QLabel#TaskLevel {
                color: #18222b;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#CardDetail {
                color: #5d6a70;
                font-size: 12px;
            }
            QComboBox {
                background: #fffaf3;
                color: #18222b;
                border: 1px solid #d6c5b0;
                border-radius: 14px;
                padding: 10px 14px;
                min-height: 22px;
            }
            QPushButton {
                border-radius: 16px;
                padding: 12px 20px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#PrimaryButton {
                background: #c65a3d;
                color: white;
            }
            QPushButton#SecondaryButton {
                background: #243847;
                color: white;
            }
            QPushButton#GhostButton {
                background: #fffaf3;
                color: #243847;
                border: 1px solid #d6c5b0;
            }
            """
        )

    def build_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        canvas = QWidget()
        canvas.setObjectName("Canvas")
        scroll.setWidget(canvas)

        root = QVBoxLayout(canvas)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("HeroPanel")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(28, 26, 28, 26)
        hero_layout.setSpacing(10)

        hero_top = QHBoxLayout()
        hero_top.setSpacing(12)

        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(6)
        hero_title = QLabel("Face AI 피부 분석 스튜디오")
        hero_title.setObjectName("HeroTitle")
        hero_title.setWordWrap(True)
        hero_subtitle = QLabel(
            "900x1440 세로 UI에서 웹캠 얼굴을 읽고 이마, 눈가, 볼, 입술, 턱 상태를 실시간으로 표시합니다."
        )
        hero_subtitle.setObjectName("HeroSubTitle")
        hero_subtitle.setWordWrap(True)
        title_wrap.addWidget(hero_title)
        title_wrap.addWidget(hero_subtitle)

        self.status_pill = QLabel("모델 로드 완료")
        self.status_pill.setObjectName("StatusPill")
        self.status_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hero_top.addLayout(title_wrap, 1)
        hero_top.addWidget(self.status_pill, 0, Qt.AlignmentFlag.AlignTop)
        hero_layout.addLayout(hero_top)

        self.session_label = QLabel(
            f"분석 장치: {self.analyzer.device} · 체크포인트: {self.analyzer.checkpoint_root.name} · 보정: {self.analyzer.use_reference_calibration}"
        )
        self.session_label.setObjectName("HeroSubTitle")
        hero_layout.addWidget(self.session_label)
        root.addWidget(hero)

        preview_panel = QFrame()
        preview_panel.setObjectName("PreviewPanel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(18, 18, 18, 18)
        preview_layout.setSpacing(14)

        preview_title = QLabel("실시간 프리뷰")
        preview_title.setObjectName("SectionTitle")
        preview_note = QLabel("정면으로 얼굴을 맞추면 부위별 박스와 상태 카드가 자동 갱신됩니다.")
        preview_note.setObjectName("SectionSubTitle")

        self.preview_label = QLabel("카메라를 준비하고 있습니다.")
        self.preview_label.setObjectName("PreviewLabel")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(640)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(preview_note)
        preview_layout.addWidget(self.preview_label)
        root.addWidget(preview_panel)

        control_panel = QFrame()
        control_panel.setObjectName("ControlPanel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(22, 20, 22, 20)
        control_layout.setSpacing(14)

        controls_title = QLabel("컨트롤")
        controls_title.setObjectName("SectionTitle")
        controls_hint = QLabel(
            "기본 카메라를 자동 시작합니다. 다른 카메라로 바꾸거나 현재 화면을 저장할 수 있습니다."
        )
        controls_hint.setObjectName("SectionSubTitle")

        row = QHBoxLayout()
        row.setSpacing(12)

        self.camera_selector = QComboBox()
        self.camera_selector.currentIndexChanged.connect(self.on_camera_changed)

        self.start_button = QPushButton("카메라 시작")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self.start_selected_camera)

        self.stop_button = QPushButton("정지")
        self.stop_button.setObjectName("SecondaryButton")
        self.stop_button.clicked.connect(self.stop_camera)

        self.snapshot_button = QPushButton("현재 화면 저장")
        self.snapshot_button.setObjectName("GhostButton")
        self.snapshot_button.clicked.connect(self.save_snapshot)

        row.addWidget(self.camera_selector, 1)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.snapshot_button)

        self.status_detail = QLabel("카메라를 찾는 중입니다.")
        self.status_detail.setObjectName("SectionSubTitle")
        self.status_detail.setWordWrap(True)

        control_layout.addWidget(controls_title)
        control_layout.addWidget(controls_hint)
        control_layout.addLayout(row)
        control_layout.addWidget(self.status_detail)
        root.addWidget(control_panel)

        summary_title = QLabel("요약 지수")
        summary_title.setObjectName("SectionTitle")
        summary_subtitle = QLabel("공식 수치가 아닌, 모델 클래스 결과를 0-100 상대 지수로 환산한 표시입니다.")
        summary_subtitle.setObjectName("SectionSubTitle")
        root.addWidget(summary_title)
        root.addWidget(summary_subtitle)

        summary_grid_widget = QWidget()
        summary_grid = QGridLayout(summary_grid_widget)
        summary_grid.setContentsMargins(0, 0, 0, 0)
        summary_grid.setHorizontalSpacing(14)
        summary_grid.setVerticalSpacing(14)
        for index, metric_name in enumerate(METRIC_META):
            card = SummaryCard(METRIC_META[metric_name]["title"], METRIC_META[metric_name]["accent"])
            self.metric_cards[metric_name] = card
            summary_grid.addWidget(card, index // 2, index % 2)
        root.addWidget(summary_grid_widget)

        detail_title = QLabel("상세 부위")
        detail_title.setObjectName("SectionTitle")
        detail_subtitle = QLabel("부위별 등급, 신뢰도, 상대 지수를 함께 확인할 수 있습니다.")
        detail_subtitle.setObjectName("SectionSubTitle")
        root.addWidget(detail_title)
        root.addWidget(detail_subtitle)

        detail_widget = QWidget()
        detail_grid = QGridLayout(detail_widget)
        detail_grid.setContentsMargins(0, 0, 0, 0)
        detail_grid.setHorizontalSpacing(14)
        detail_grid.setVerticalSpacing(14)

        for index, task_name in enumerate(TASK_META):
            card = TaskCard(TASK_META[task_name]["title"], TASK_META[task_name]["accent"])
            self.task_cards[task_name] = card
            detail_grid.addWidget(card, index // 2, index % 2)
        root.addWidget(detail_widget)

        footer = QLabel(
            "안내: 이 UI는 공개 체크포인트를 웹캠 입력에 맞게 변환해 실행한 실시간 추정 화면입니다. 검증용 bbox JSON을 직접 쓰는 공식 평가와는 입력 조건이 다릅니다."
        )
        footer.setObjectName("SectionSubTitle")
        footer.setWordWrap(True)
        root.addWidget(footer)

        root.addStretch(1)

    def populate_camera_selector(self) -> None:
        current_index = self.args.camera_index
        indices = list(range(max(1, self.args.max_cameras)))
        if current_index not in indices:
            indices.insert(0, current_index)

        self.camera_selector.blockSignals(True)
        self.camera_selector.clear()
        for index in indices:
            self.camera_selector.addItem(f"Camera {index}", index)
        matched_row = max(0, self.camera_selector.findData(current_index))
        self.camera_selector.setCurrentIndex(matched_row)
        self.camera_selector.blockSignals(False)

    def open_camera(self, index: int):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            return capture

        capture.release()
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)

    def selected_camera_index(self) -> int:
        data = self.camera_selector.currentData()
        return int(data) if data is not None else int(self.args.camera_index)

    def start_selected_camera(self) -> None:
        self.stop_camera()
        camera_index = self.selected_camera_index()
        capture = self.open_camera(camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, 30)

        if not capture.isOpened():
            self.capture = None
            self.set_status("카메라 연결 실패", f"Camera {camera_index} 를 열지 못했습니다.")
            self.preview_label.setText("카메라를 열 수 없습니다.")
            return

        self.capture = capture
        self.timer.start()
        self.set_status("실시간 분석 중", f"Camera {camera_index} 를 사용 중입니다.")

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.set_status("카메라 정지", "원할 때 다시 시작할 수 있습니다.")

    def on_camera_changed(self) -> None:
        if self.capture is not None:
            self.start_selected_camera()

    def set_status(self, short_text: str, detail_text: str) -> None:
        self.status_pill.setText(short_text)
        self.status_detail.setText(detail_text)

    def update_frame(self) -> None:
        if self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self.set_status("프레임 읽기 실패", "카메라 프레임을 다시 요청하고 있습니다.")
            return

        self.latest_frame = frame.copy()
        display_frame = draw_analysis_overlay(frame, self.latest_result or {"face_detected": False, "message": "얼굴을 감지하는 중입니다."})
        self.update_preview(display_frame)

        if self.inference_busy:
            return

        now = time.monotonic()
        if now - self.last_inference_time < self.args.inference_interval:
            return

        self.inference_busy = True
        self.last_inference_time = now
        self.inference_thread = InferenceThread(self.analyzer, frame.copy())
        self.inference_thread.result_ready.connect(self.on_inference_result)
        self.inference_thread.failed.connect(self.on_inference_failed)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.start()

    def on_inference_result(self, result: dict) -> None:
        self.latest_result = result

        if result.get("face_detected"):
            overall_score = result.get("overall_score")
            overall_label = result.get("overall_label")
            self.set_status(
                "피부 상태 분석 중",
                f"마지막 업데이트 {datetime.now().strftime('%H:%M:%S')} · 전체 상대 지수 {overall_score:.1f} ({overall_label})"
                if overall_score is not None
                else "마지막 분석이 완료되었습니다.",
            )
        else:
            self.set_status("얼굴 감지 대기", result.get("message", "얼굴을 프레임 중앙에 맞춰 주세요."))

        for metric_name, card in self.metric_cards.items():
            card.update_metric(result.get("metrics", {}).get(metric_name))

        for task_name, card in self.task_cards.items():
            card.update_task(result.get("tasks", {}).get(task_name))

    def on_inference_failed(self, message: str) -> None:
        self.set_status("분석 오류", message)

    def on_inference_finished(self) -> None:
        self.inference_busy = False
        if self.inference_thread is not None:
            self.inference_thread.deleteLater()
            self.inference_thread = None

    def update_preview(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        target = self.preview_label.size()
        scaled = pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def save_snapshot(self) -> None:
        if self.latest_frame is None:
            self.set_status("저장 실패", "아직 저장할 프레임이 없습니다.")
            return

        annotated = draw_analysis_overlay(self.latest_frame, self.latest_result or {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.snapshot_dir / f"face_ai_live_{timestamp}.jpg"
        save_image_unicode(output_path, annotated)
        self.set_status("현재 화면 저장", f"{output_path} 에 저장했습니다.")

    def closeEvent(self, event) -> None:
        self.stop_camera()
        if self.inference_thread is not None and self.inference_thread.isRunning():
            self.inference_thread.quit()
            self.inference_thread.wait(1500)
        super().closeEvent(event)


def parse_args():
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(
        description="Launch the Face_AI live webcam UI built with PySide6."
    )
    parser.add_argument("--camera-index", default=0, type=int)
    parser.add_argument("--window-width", default=900, type=int)
    parser.add_argument("--window-height", default=1440, type=int)
    parser.add_argument("--max-cameras", default=5, type=int)
    parser.add_argument("--inference-interval", default=0.7, type=float)
    parser.add_argument("--checkpoint-root", default=str(defaults["checkpoint_root"]))
    parser.add_argument("--snapshot-dir", default=str(defaults["snapshot_dir"]))
    parser.add_argument("--disable-reference-calibration", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Malgun Gothic", 10))
    window = FaceAILiveWindow(args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
