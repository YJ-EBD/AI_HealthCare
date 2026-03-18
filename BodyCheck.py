from __future__ import annotations

import argparse
import ctypes
import math
import signal
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - depends on local environment
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    from PyQt5.QtCore import QEasingCurve, QParallelAnimationGroup, QPropertyAnimation, Qt, QRect, QTimer
    from PyQt5.QtGui import QFont, QImage, QPainter, QPen, QPixmap
    from PyQt5.QtWidgets import QApplication, QCheckBox, QDialog, QLabel, QPushButton, QStyle, QStyleOptionButton, QWidget
except ImportError:  # pragma: no cover - depends on local environment
    QApplication = None
    QCheckBox = None
    QDialog = None
    QLabel = None
    QPushButton = None
    QPixmap = None
    QFont = None
    QPainter = None
    QPen = None
    QStyle = None
    QStyleOptionButton = None
    QWidget = None
    QEasingCurve = None
    QParallelAnimationGroup = None
    QPropertyAnimation = None
    QRect = None
    QTimer = None
    QImage = None
    Qt = None

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "mediapipe is required for BodyCheck.py.\n"
        "Install with: .\\.venv\\Scripts\\python.exe -m pip install mediapipe"
    ) from exc


Color = Tuple[int, int, int]
WINDOW_NAME = "Body Check"
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}
ROTATION_FLAGS = {
    "none": None,
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
}
ROTATION_ORDER = ("none", "cw", "ccw")
COMMON_CAPTURE_MODES = (
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (640, 480),
)
UI_DIR = Path(__file__).resolve().parent / "UI"
MAIN_LAYOUT_PATH = UI_DIR / "Main.png"
QNA_LAYOUT_PATH = UI_DIR / "QnA.png"
TYPEA_LAYOUT_PATH = UI_DIR / "TypeA.png"
# Approximate search box for the inner rounded camera area in UI/TypeA.png.
BODY_CAMERA_RECT = (102, 141, 485, 818)
FACE_A_CAMERA_RECT = (676, 139, 549, 325)
FACE_B_CAMERA_RECT = (1296, 139, 549, 325)
MAIN_LOGO_PADDING_X_RATIO = 0.014
MAIN_LOGO_PADDING_Y_RATIO = 0.02
MAIN_LOGO_FALLBACK_WIDTH_RATIO = 0.18
MAIN_LOGO_FALLBACK_HEIGHT_RATIO = 0.12
BODY_CAMERA_MARGIN = 0
BODY_CAMERA_RADIUS = 23
FACE_CAMERA_RADIUS = 26
SURVEY_BUTTON_TEXT = "설문완료"
SURVEY_BUTTON_WIDTH_RATIO = 0.075
SURVEY_BUTTON_HEIGHT_RATIO = 0.046
SURVEY_BUTTON_RIGHT_MARGIN_RATIO = 0.012
SURVEY_BUTTON_BOTTOM_MARGIN_RATIO = 0.018
SURVEY_BUTTON_RADIUS_RATIO = 0.45
SURVEY_BUTTON_SAFE_GAP_RATIO = 0.004
SURVEY_TITLE_FONT_SIZE = 24
SURVEY_OPTION_FONT_SIZE = 18
SURVEY_TEXT_RENDER_SCALE = 4
TONGUE_PREVIEW_HOLD_FRAMES = 8
SURVEY_TEXT_COLOR: Color = (248, 251, 255)
OVERLAY_TEXT_PRIMARY: Color = (246, 249, 255)
OVERLAY_TEXT_SECONDARY: Color = (220, 228, 240)
OVERLAY_TEXT_GUIDE_READY: Color = (240, 248, 242)
OVERLAY_TEXT_GUIDE_WAIT: Color = (236, 242, 250)
OVERLAY_TEXT_METRIC_COLORS: Dict[str, Color] = {
    "good": (243, 250, 245),
    "warn": (255, 248, 242),
    "bad": (255, 242, 242),
}


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    visibility: float = 1.0


@dataclass(frozen=True)
class MetricSpec:
    label: str
    good_max: float
    warn_max: float
    unit: str
    description: str


@dataclass(frozen=True)
class MetricReading:
    label: str
    value: float
    unit: str
    status: str
    color: Color
    description: str


@dataclass
class PoseDetection:
    landmarks: Optional[list]
    raw_result: object = None


@dataclass
class UiState:
    current_page: str = "survey"
    survey_button_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    survey_button_hovered: bool = False
    survey_completed: bool = False
    survey_button_accent: Color = (242, 196, 160)
    survey_option_states: list[bool] = field(default_factory=list)
    survey_hovered_option: Optional[int] = None
    survey_base_frame: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SurveySectionSpec:
    title: str
    options: Tuple[str, ...]
    title_rect: Tuple[int, int, int, int]
    option_rects: Tuple[Tuple[int, int, int, int], ...]
    checkbox_rects: Tuple[Tuple[int, int, int, int], ...]
    click_rects: Tuple[Tuple[int, int, int, int], ...]


SURVEY_LEFT_SECTIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "소양인",
        (
            "성격이 급하고 추진력이 강함",
            "더위에 약하고 땀이 많음",
            "얼굴/상체에 열이 잘 오름",
            "잠이 얕고 예민함",
        ),
    ),
    (
        "태음인",
        (
            "체격이 크거나 살이 잘 찜",
            "참을성이 많고 느긋함",
            "땀을 흘리면 개운함",
            "호흡기/피로/비만 경향",
        ),
    ),
    (
        "소음인",
        (
            "손발이 차고 추위를 탐",
            "소화가 약하고 설사/복통",
            "신중하고 걱정이 많음",
            "적게 먹어도 배부름",
        ),
    ),
    (
        "태양인 (희귀)",
        (
            "가슴/어깨 발달",
            "하체/복부 약함",
            "리더형/독립적",
            "소변/피로 문제",
        ),
    ),
)

SURVEY_RIGHT_SECTIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "열 성향",
        (
            "얼굴이 자주 붉어짐",
            "입 마름, 불면",
            "염증/트러블 잦음",
            "",
        ),
    ),
    (
        "냉 성향",
        (
            "손발 차가움",
            "따뜻한 음식 선호",
            "복부 냉감",
            "",
        ),
    ),
    (
        "허 성향",
        (
            "쉽게 피곤",
            "숨참, 기력 부족",
            "회복 느림",
            "",
        ),
    ),
    (
        "담/정체 성향",
        (
            "몸이 무겁다",
            "가래/부종",
            "눌렀을 때 단단함",
            "",
        ),
    ),
)


METRIC_SPECS: Dict[str, MetricSpec] = {
    "shoulder_tilt": MetricSpec(
        label="Shoulder Tilt",
        good_max=3.0,
        warn_max=6.0,
        unit="deg",
        description="Difference in shoulder line height.",
    ),
    "pelvis_tilt": MetricSpec(
        label="Pelvis / Waist Tilt",
        good_max=3.0,
        warn_max=6.0,
        unit="deg",
        description="Difference in hip line height.",
    ),
    "torso_lean": MetricSpec(
        label="Torso Lean",
        good_max=4.0,
        warn_max=8.0,
        unit="deg",
        description="How much the trunk leans from vertical.",
    ),
    "center_shift": MetricSpec(
        label="Center Shift",
        good_max=0.05,
        warn_max=0.09,
        unit="ratio",
        description="Shoulder center drifting away from hip center.",
    ),
}

STATUS_COLORS: Dict[str, Color] = {
    "good": (60, 185, 90),
    "warn": (0, 190, 255),
    "bad": (0, 90, 255),
}


def midpoint(a: Point, b: Point) -> Point:
    return Point(
        x=(a.x + b.x) / 2.0,
        y=(a.y + b.y) / 2.0,
        visibility=min(a.visibility, b.visibility),
    )


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def horizontal_angle_deg(left: Point, right: Point) -> float:
    return abs(math.degrees(math.atan2(right.y - left.y, right.x - left.x)))


def vertical_lean_deg(top: Point, bottom: Point) -> float:
    dx = top.x - bottom.x
    dy = top.y - bottom.y
    return abs(math.degrees(math.atan2(dx, dy if abs(dy) > 1e-6 else 1e-6)))


def classify_metric(metric_name: str, value: float) -> MetricReading:
    spec = METRIC_SPECS[metric_name]
    if value <= spec.good_max:
        status = "good"
    elif value <= spec.warn_max:
        status = "warn"
    else:
        status = "bad"
    return MetricReading(
        label=spec.label,
        value=value,
        unit=spec.unit,
        status=status,
        color=STATUS_COLORS[status],
        description=spec.description,
    )


def portrait_display_size(width: int, height: int) -> Tuple[int, int]:
    return min(width, height), max(width, height)


def next_rotation(rotation: str) -> str:
    index = ROTATION_ORDER.index(rotation)
    return ROTATION_ORDER[(index + 1) % len(ROTATION_ORDER)]


def apply_rotation(frame: np.ndarray, rotation: str) -> np.ndarray:
    rotate_flag = ROTATION_FLAGS[rotation]
    if rotate_flag is None:
        return frame
    return cv2.rotate(frame, rotate_flag)


def fit_frame_to_canvas(frame: np.ndarray, canvas_width: int, canvas_height: int) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    scale = max(canvas_width / frame_width, canvas_height / frame_height)
    new_width = max(1, int(frame_width * scale))
    new_height = max(1, int(frame_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

    start_x = max(0, (new_width - canvas_width) // 2)
    start_y = max(0, (new_height - canvas_height) // 2)
    return resized[start_y : start_y + canvas_height, start_x : start_x + canvas_width].copy()


def fit_frame_to_canvas_contain(
    frame: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    if background is None:
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    else:
        canvas = background.copy()

    frame_height, frame_width = frame.shape[:2]
    scale = min(canvas_width / frame_width, canvas_height / frame_height)
    new_width = max(1, int(frame_width * scale))
    new_height = max(1, int(frame_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

    offset_x = max(0, (canvas_width - new_width) // 2)
    offset_y = max(0, (canvas_height - new_height) // 2)
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized
    return canvas


def preferred_capture_modes(width: int, height: int) -> Tuple[Tuple[int, int], ...]:
    requested = (width, height) if width >= height else (height, width)
    modes = [requested]
    for mode in COMMON_CAPTURE_MODES:
        if mode not in modes:
            modes.append(mode)
    return tuple(modes)


def configure_capture_mode(cap: cv2.VideoCapture, width: int, height: int) -> Tuple[int, int]:
    for mode_width, mode_height in preferred_capture_modes(width, height):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode_height)
        ok, frame = cap.read()
        if not ok:
            continue

        actual_height, actual_width = frame.shape[:2]
        if actual_width >= min(mode_width, mode_height) and actual_height >= min(mode_width, mode_height) * 0.5:
            return actual_width, actual_height

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Unable to read an initial frame from the webcam.")
    actual_height, actual_width = frame.shape[:2]
    return actual_width, actual_height


def load_ui_layout(layout_path: Path) -> Optional[np.ndarray]:
    if not layout_path.exists():
        return None

    layout = cv2.imread(str(layout_path))
    if layout is None:
        raise RuntimeError(f"Failed to load UI layout image: {layout_path}")
    return layout


def set_window_arrow_cursor(window_name: str) -> None:
    if not sys.platform.startswith("win"):
        return

    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return

        cursor = user32.LoadCursorW(None, 32512)  # IDC_ARROW
        if not cursor:
            return

        gclp_hcursor = -12
        if hasattr(user32, "SetClassLongPtrW"):
            user32.SetClassLongPtrW(hwnd, gclp_hcursor, cursor)
        else:
            user32.SetClassLongW(hwnd, gclp_hcursor, cursor)
        user32.SetCursor(cursor)
        user32.SendMessageW(hwnd, 0x20, hwnd, 1)  # WM_SETCURSOR / HTCLIENT
    except Exception:
        return


def set_window_fullscreen_borderless(window_name: str) -> None:
    if sys.platform.startswith("win"):
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, window_name)
            if not hwnd:
                return

            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
            gwi_style = -16
            ws_visible = 0x10000000
            ws_popup = 0x80000000
            swp_framechanged = 0x0020
            swp_showwindow = 0x0040

            user32.SetWindowLongW(hwnd, gwi_style, ws_visible | ws_popup)
            user32.SetWindowPos(
                hwnd,
                0,
                0,
                0,
                screen_width,
                screen_height,
                swp_framechanged | swp_showwindow,
            )
            if hasattr(cv2, "WND_PROP_FULLSCREEN") and hasattr(cv2, "WINDOW_FULLSCREEN"):
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            return
        except Exception:
            pass

    if hasattr(cv2, "WND_PROP_FULLSCREEN") and hasattr(cv2, "WINDOW_FULLSCREEN"):
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except cv2.error:
            return


def clamp_color(color: Iterable[float]) -> Color:
    return tuple(max(0, min(255, int(round(channel)))) for channel in color)


def blend_colors(base: Color, target: Color, weight: float) -> Color:
    weight = max(0.0, min(1.0, weight))
    return clamp_color(
        (base[index] * (1.0 - weight)) + (target[index] * weight)
        for index in range(3)
    )


def estimate_layout_accent_color(layout: np.ndarray) -> Color:
    hsv = cv2.cvtColor(layout, cv2.COLOR_BGR2HSV)
    saturation_mask = hsv[:, :, 1] > 40
    value_mask = hsv[:, :, 2] > 120
    mask = saturation_mask & value_mask
    if not np.any(mask):
        return (242, 196, 160)

    pixels = layout[mask]
    mean_color = np.mean(pixels, axis=0)
    return clamp_color(mean_color)


def estimate_survey_content_right_edge(background: np.ndarray) -> int:
    canvas_height, canvas_width = background.shape[:2]
    scan_top = int(canvas_height * 0.8)
    region = background[scan_top:, :]
    if region.size == 0:
        return int(canvas_width * 0.9)

    active_mask = np.any(region < 245, axis=2)
    min_pixels = max(6, int(region.shape[0] * 0.08))
    active_columns = np.where(active_mask.sum(axis=0) > min_pixels)[0]
    if active_columns.size == 0:
        return int(canvas_width * 0.9)
    return int(active_columns.max())


def compute_survey_button_rect(background: np.ndarray) -> Tuple[int, int, int, int]:
    canvas_height, canvas_width = background.shape[:2]
    margin_right = max(18, int(canvas_width * SURVEY_BUTTON_RIGHT_MARGIN_RATIO))
    margin_bottom = max(16, int(canvas_height * SURVEY_BUTTON_BOTTOM_MARGIN_RATIO))
    safe_gap = max(8, int(canvas_width * SURVEY_BUTTON_SAFE_GAP_RATIO))
    preferred_width = max(120, int(canvas_width * SURVEY_BUTTON_WIDTH_RATIO))
    button_height = max(46, int(canvas_height * SURVEY_BUTTON_HEIGHT_RATIO))
    content_right_edge = estimate_survey_content_right_edge(background)
    max_safe_width = canvas_width - content_right_edge - safe_gap - margin_right
    button_width = min(preferred_width, max(110, max_safe_width))
    button_x = canvas_width - button_width - margin_right
    button_y = canvas_height - button_height - margin_bottom
    return button_x, button_y, button_width, button_height


def compute_main_logo_rect(background: np.ndarray) -> Tuple[int, int, int, int]:
    canvas_height, canvas_width = background.shape[:2]
    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 9)
    detail = cv2.absdiff(gray, blurred)
    _, detail_mask = cv2.threshold(detail, 14, 255, cv2.THRESH_BINARY)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(detail_mask)
    min_area = max(250, int(canvas_width * canvas_height * 0.00005))
    best_rect: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0

    for component_index in range(1, component_count):
        x, y, width, height, area = stats[component_index]
        if area < min_area:
            continue
        if x < int(canvas_width * 0.55) or y < int(canvas_height * 0.55):
            continue

        score = float(area) + (x * 0.4) + (y * 0.4)
        if score > best_score:
            best_score = score
            best_rect = (int(x), int(y), int(width), int(height))

    if best_rect is None:
        fallback_width = max(240, int(canvas_width * MAIN_LOGO_FALLBACK_WIDTH_RATIO))
        fallback_height = max(96, int(canvas_height * MAIN_LOGO_FALLBACK_HEIGHT_RATIO))
        fallback_x = canvas_width - fallback_width - max(36, int(canvas_width * 0.03))
        fallback_y = canvas_height - fallback_height - max(28, int(canvas_height * 0.03))
        return fallback_x, fallback_y, fallback_width, fallback_height

    x, y, width, height = best_rect
    padding_x = max(18, int(canvas_width * MAIN_LOGO_PADDING_X_RATIO))
    padding_y = max(14, int(canvas_height * MAIN_LOGO_PADDING_Y_RATIO))
    left = max(0, x - padding_x)
    top = max(0, y - padding_y)
    right = min(canvas_width, x + width + padding_x)
    bottom = min(canvas_height, y + height + padding_y)
    return left, top, max(1, right - left), max(1, bottom - top)


def build_survey_section_specs() -> Tuple[SurveySectionSpec, ...]:
    title_x_positions = (225, 1085)
    option_x_positions = (268, 1128)
    checkbox_x_positions = (231, 1091)
    section_title_y_positions = (92, 311, 530, 749)
    option_box_offsets = (62, 100, 138, 176)
    checkbox_offsets = (69, 107, 145, 183)
    title_size = (582, 53)
    option_size = (539, 36)
    checkbox_size = (22, 22)
    section_columns = (SURVEY_LEFT_SECTIONS, SURVEY_RIGHT_SECTIONS)

    sections: list[SurveySectionSpec] = []
    for column_index, column_sections in enumerate(section_columns):
        title_x = title_x_positions[column_index]
        option_x = option_x_positions[column_index]
        checkbox_x = checkbox_x_positions[column_index]
        for row_index, (title, options) in enumerate(column_sections):
            title_y = section_title_y_positions[row_index]
            title_rect = (title_x, title_y, title_size[0], title_size[1])
            option_rects: list[Tuple[int, int, int, int]] = []
            checkbox_rects: list[Tuple[int, int, int, int]] = []
            click_rects: list[Tuple[int, int, int, int]] = []
            for option_offset, checkbox_offset in zip(option_box_offsets, checkbox_offsets):
                option_y = title_y + option_offset
                checkbox_y = title_y + checkbox_offset
                option_rect = (option_x, option_y, option_size[0], option_size[1])
                checkbox_rect = (checkbox_x, checkbox_y, checkbox_size[0], checkbox_size[1])
                click_rect = (checkbox_x - 6, option_y, (option_x + option_size[0]) - (checkbox_x - 6), option_size[1])
                option_rects.append(option_rect)
                checkbox_rects.append(checkbox_rect)
                click_rects.append(click_rect)

            sections.append(
                SurveySectionSpec(
                    title=title,
                    options=options,
                    title_rect=title_rect,
                    option_rects=tuple(option_rects),
                    checkbox_rects=tuple(checkbox_rects),
                    click_rects=tuple(click_rects),
                )
            )
    return tuple(sections)


SURVEY_SECTION_SPECS = build_survey_section_specs()
SURVEY_OPTION_COUNT = sum(len(section.options) for section in SURVEY_SECTION_SPECS)


if QDialog is not None:

    class TransparentSurveyCheckBox(QCheckBox):
        def paintEvent(self, event) -> None:
            super().paintEvent(event)
            if not self.isChecked() or QPainter is None or QPen is None or QStyle is None or QStyleOptionButton is None:
                return

            option = QStyleOptionButton()
            self.initStyleOption(option)
            indicator_rect = self.style().subElementRect(QStyle.SE_CheckBoxIndicator, option, self)

            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen()
            pen.setColor(Qt.white)
            pen.setWidth(2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)

            inset = max(1, indicator_rect.width() // 6)
            left = indicator_rect.left() + inset
            top = indicator_rect.top() + inset
            right = indicator_rect.right() - inset
            bottom = indicator_rect.bottom() - inset
            mid_x = left + int((right - left) * 0.35)
            mid_y = top + int((bottom - top) * 0.65)
            check_points = [
                (left, top + int((bottom - top) * 0.55)),
                (mid_x, bottom),
                (right, top),
            ]
            for start, end in zip(check_points, check_points[1:]):
                painter.drawLine(start[0], start[1], end[0], end[1])

    class SurveyDialog(QDialog):
        def __init__(
            self,
            main_background_path: Path,
            main_logo_rect: Tuple[int, int, int, int],
            survey_background_path: Path,
            survey_button_rect: Tuple[int, int, int, int],
            analysis_layout: np.ndarray,
            args: argparse.Namespace,
            initial_states: Optional[list[bool]] = None,
        ) -> None:
            super().__init__()
            self._base_size = (1920, 1080)
            self._main_background_pixmap = QPixmap(str(main_background_path))
            self._survey_background_pixmap = QPixmap(str(survey_background_path))
            self._main_logo_rect = main_logo_rect
            self._survey_button_rect = survey_button_rect
            self._analysis_layout = analysis_layout
            self._args = args
            self._option_states = list(initial_states or ([False] * SURVEY_OPTION_COUNT))
            self._titles: list[Tuple[QLabel, Tuple[int, int, int, int]]] = []
            self._checkboxes: list[Tuple[QCheckBox, Tuple[int, int, int, int]]] = []
            self._transition_group = None
            self._transition_from_page: Optional[QWidget] = None
            self._transition_to_page: Optional[QWidget] = None
            self._transition_target: Optional[str] = None
            self._after_transition = None
            self._current_page = "main"
            self._analysis_frame: Optional[np.ndarray] = None
            self._summary_text: Optional[str] = None
            self._backend = None
            self._cap = None
            self._analyzer = None
            self._face_detector = None
            self._smoother: Optional[MetricSmoother] = None
            self._last_tongue_preview: Optional[np.ndarray] = None
            self._tongue_preview_hold_frames = 0
            self._flip_view = not args.no_flip
            self._rotation = args.rotation
            self._frame_timer = QTimer(self)
            self._frame_timer.setInterval(15)
            self._frame_timer.timeout.connect(self._update_analysis_frame)

            self._ui_body_target = resolve_body_camera_target(self._analysis_layout)
            self._ui_face_target = resolve_face_camera_target(self._analysis_layout, FACE_A_CAMERA_RECT)
            self._ui_tongue_target = resolve_face_camera_target(self._analysis_layout, FACE_B_CAMERA_RECT)
            self._face_preview_aspect_ratio = self._ui_face_target[0][2] / max(1.0, self._ui_face_target[0][3])
            self._tongue_preview_aspect_ratio = self._ui_tongue_target[0][2] / max(1.0, self._ui_tongue_target[0][3])
            self._loading_frame = compose_loading_ui_frame(
                self._analysis_layout,
                "--",
                self._ui_body_target,
                self._ui_face_target,
                self._ui_tongue_target,
            )

            self.setWindowTitle(WINDOW_NAME)
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
            self.setObjectName("SurveyDialog")
            self.setStyleSheet("#SurveyDialog { background-color: #f7f7f7; }")

            self._main_page = QWidget(self)
            self._survey_page = QWidget(self)
            self._analysis_page = QWidget(self)
            self._pages = {
                "main": self._main_page,
                "survey": self._survey_page,
                "analysis": self._analysis_page,
            }

            self._main_background_label = QLabel(self._main_page)
            self._main_background_label.setScaledContents(True)
            self._main_background_label.lower()

            self._main_logo_button = QPushButton("", self._main_page)
            self._main_logo_button.clicked.connect(self._start_intro_transition)
            self._main_logo_button.setCursor(Qt.PointingHandCursor)
            self._main_logo_button.setStyleSheet(
                """
                QPushButton {
                    background: transparent;
                    border: none;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 28px;
                }
                QPushButton:pressed {
                    background-color: rgba(255, 255, 255, 0.1);
                    border-radius: 28px;
                }
                """
            )

            self._survey_background_label = QLabel(self._survey_page)
            self._survey_background_label.setScaledContents(True)
            self._survey_background_label.lower()

            self._analysis_label = QLabel(self._analysis_page)
            self._analysis_label.setScaledContents(True)
            self._analysis_label.lower()

            for section in SURVEY_SECTION_SPECS:
                title_label = QLabel(section.title, self._survey_page)
                title_label.setAlignment(Qt.AlignCenter)
                title_label.setStyleSheet("background: transparent; color: #f8fbff;")
                self._titles.append((title_label, section.title_rect))

                for option_text, option_rect, checkbox_rect in zip(
                    section.options,
                    section.option_rects,
                    section.checkbox_rects,
                ):
                    if not option_text:
                        continue
                    checkbox_row_rect = (
                        checkbox_rect[0],
                        option_rect[1],
                        (option_rect[0] + option_rect[2]) - checkbox_rect[0],
                        option_rect[3],
                    )
                    checkbox = TransparentSurveyCheckBox(option_text, self._survey_page)
                    checkbox.setStyleSheet(
                        """
                        QCheckBox {
                            background: transparent;
                            color: #f8fbff;
                            spacing: 23px;
                            padding: 0px;
                        }
                        QCheckBox::indicator {
                            width: 18px;
                            height: 18px;
                            background: transparent;
                            border: none;
                        }
                        QCheckBox::indicator:unchecked {
                            background: transparent;
                            border: none;
                            image: none;
                        }
                        QCheckBox::indicator:checked {
                            background: transparent;
                            border: none;
                            image: none;
                        }
                        """
                    )
                    self._checkboxes.append((checkbox, checkbox_row_rect))

            self._complete_button = QPushButton(SURVEY_BUTTON_TEXT, self._survey_page)
            self._complete_button.clicked.connect(self._start_survey_transition)
            self._complete_button.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(116, 139, 193, 0.97);
                    color: #f8fbff;
                    border: 2px solid rgba(183, 202, 238, 0.96);
                    border-radius: 26px;
                    padding: 2px 10px;
                }
                QPushButton:hover {
                    background-color: rgba(129, 152, 207, 0.98);
                }
                QPushButton:pressed {
                    background-color: rgba(103, 126, 181, 0.98);
                }
                """
            )

            self._show_only_page("main")

        def option_states(self) -> list[bool]:
            return [checkbox.isChecked() for checkbox, _ in self._checkboxes]

        def final_summary(self) -> Optional[str]:
            return self._summary_text

        def _show_only_page(self, page_name: str) -> None:
            for name, page in self._pages.items():
                page.setGeometry(self.rect())
                if name == page_name:
                    page.show()
                    page.raise_()
                else:
                    page.hide()

        def _set_analysis_frame(self, frame: np.ndarray) -> None:
            self._analysis_frame = frame.copy()
            if QImage is None:
                return

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb_frame.shape[:2]
            qimage = QImage(rgb_frame.data, width, height, rgb_frame.strides[0], QImage.Format_RGB888)
            target_size = self._analysis_label.size()
            if target_size.width() <= 0 or target_size.height() <= 0:
                target_size = self.size()
            pixmap = QPixmap.fromImage(qimage.copy())
            self._analysis_label.setPixmap(
                pixmap.scaled(
                    target_size,
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        def _start_intro_transition(self) -> None:
            self._start_slide_transition("survey")

        def _start_survey_transition(self) -> None:
            self._option_states = self.option_states()
            self._set_analysis_frame(self._loading_frame)
            self._start_slide_transition("analysis", after_transition=self._start_analysis)

        def _start_slide_transition(self, next_page_name: str, after_transition=None) -> None:
            if self._transition_group is not None or next_page_name == self._current_page:
                return

            current_page = self._pages[self._current_page]
            next_page = self._pages[next_page_name]
            self._after_transition = after_transition

            if (
                QParallelAnimationGroup is None
                or QPropertyAnimation is None
                or QRect is None
                or QEasingCurve is None
            ):
                self._current_page = next_page_name
                self._show_only_page(next_page_name)
                if self._after_transition is not None:
                    callback = self._after_transition
                    self._after_transition = None
                    QTimer.singleShot(0, callback)
                return

            width = self.width()
            height = self.height()

            current_page.setGeometry(0, 0, width, height)
            next_page.setGeometry(width, 0, width, height)
            next_page.show()
            next_page.raise_()

            current_animation = QPropertyAnimation(current_page, b"geometry", self)
            current_animation.setDuration(340)
            current_animation.setStartValue(QRect(0, 0, width, height))
            current_animation.setEndValue(QRect(-width, 0, width, height))
            current_animation.setEasingCurve(QEasingCurve.OutCubic)

            next_animation = QPropertyAnimation(next_page, b"geometry", self)
            next_animation.setDuration(340)
            next_animation.setStartValue(QRect(width, 0, width, height))
            next_animation.setEndValue(QRect(0, 0, width, height))
            next_animation.setEasingCurve(QEasingCurve.OutCubic)

            self._transition_from_page = current_page
            self._transition_to_page = next_page
            self._transition_target = next_page_name
            self._transition_group = QParallelAnimationGroup(self)
            self._transition_group.addAnimation(current_animation)
            self._transition_group.addAnimation(next_animation)
            self._transition_group.finished.connect(self._finish_slide_transition)
            self._transition_group.start()

        def _finish_slide_transition(self) -> None:
            if self._transition_from_page is not None:
                self._transition_from_page.hide()
                self._transition_from_page.setGeometry(self.rect())

            if self._transition_to_page is not None:
                self._transition_to_page.setGeometry(self.rect())
                self._transition_to_page.show()
                self._transition_to_page.raise_()

            if self._transition_target is not None:
                self._current_page = self._transition_target

            callback = self._after_transition
            self._after_transition = None
            self._transition_group = None
            self._transition_from_page = None
            self._transition_to_page = None
            self._transition_target = None

            if callback is not None:
                QTimer.singleShot(0, callback)

        def _start_analysis(self) -> None:
            try:
                self._backend = create_pose_backend(
                    model_variant=self._args.model_variant,
                    model_path=self._args.model_path,
                )
                self._smoother = MetricSmoother(METRIC_SPECS.keys(), max(3, self._args.smooth))
                self._analyzer = BodyPostureAnalyzer(
                    self._backend.landmark_enum,
                    min_visibility=self._args.min_visibility,
                )
                self._face_detector = create_face_detector()
                self._cap, capture_size = open_camera(
                    self._args.camera,
                    self._args.width,
                    self._args.height,
                )
                print(
                    f"Survey complete. Starting posture analysis with {self._backend.backend_name}. "
                    f"Camera capture: {capture_size[0]}x{capture_size[1]}"
                )
                self._frame_timer.start()
            except Exception as exc:
                print(str(exc))
                self.reject()

        def _update_analysis_frame(self) -> None:
            if self._cap is None or self._backend is None or self._analyzer is None or self._smoother is None:
                return

            ok, frame = self._cap.read()
            if not ok:
                print("Failed to read from webcam.")
                self.reject()
                return

            frame = apply_rotation(frame, self._rotation)
            if self._flip_view:
                frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()

            timestamp_ms = int(time.monotonic() * 1000)
            detection = self._backend.detect(frame, timestamp_ms)

            readings: Optional[Dict[str, MetricReading]] = None
            guidance = "Stand naturally facing the camera with your full body visible."
            detection_ok = False

            metrics, message = self._analyzer.analyze(detection.landmarks)
            if metrics is not None:
                detection_ok = True
                averaged = self._smoother.update(metrics)
                readings = {
                    metric_name: classify_metric(metric_name, metric_value)
                    for metric_name, metric_value in averaged.items()
                }
                guidance = "Stable reading. Relax your arms and keep both feet on the floor."
            else:
                guidance = message or guidance

            face_preview = crop_face_region(
                raw_frame,
                self._face_detector,
                target_aspect_ratio=self._face_preview_aspect_ratio,
            )
            tongue_preview = None
            if face_preview is not None:
                tongue_detected, _tongue_score, mouth_preview = detect_tongue_in_face(
                    face_preview,
                    preview_aspect_ratio=self._tongue_preview_aspect_ratio,
                )
                if tongue_detected and mouth_preview is not None:
                    self._last_tongue_preview = mouth_preview.copy()
                    tongue_preview = mouth_preview
                    self._tongue_preview_hold_frames = TONGUE_PREVIEW_HOLD_FRAMES
                elif self._last_tongue_preview is not None and self._tongue_preview_hold_frames > 0:
                    tongue_preview = self._last_tongue_preview
                    self._tongue_preview_hold_frames -= 1
                else:
                    self._last_tongue_preview = None
                    self._tongue_preview_hold_frames = 0
            else:
                self._last_tongue_preview = None
                self._tongue_preview_hold_frames = 0

            if detection.landmarks:
                self._backend.draw(frame, detection)
                draw_reference_lines(frame, detection.landmarks, self._backend.landmark_enum)

            display_frame = compose_ui_frame(
                self._analysis_layout,
                frame,
                face_preview,
                tongue_preview,
                readings,
                guidance,
                detection_ok,
                self._ui_body_target,
                self._ui_face_target,
                self._ui_tongue_target,
            )
            self._set_analysis_frame(display_frame)

        def _scale_rect(self, rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
            base_width, base_height = self._base_size
            scale_x = self.width() / max(1.0, float(base_width))
            scale_y = self.height() / max(1.0, float(base_height))
            x, y, width, height = rect
            return (
                int(round(x * scale_x)),
                int(round(y * scale_y)),
                max(1, int(round(width * scale_x))),
                max(1, int(round(height * scale_y))),
            )

        def _apply_layout(self) -> None:
            if self._transition_group is None:
                for page in self._pages.values():
                    page.setGeometry(self.rect())

            self._main_background_label.setGeometry(self._main_page.rect())
            if not self._main_background_pixmap.isNull():
                self._main_background_label.setPixmap(
                    self._main_background_pixmap.scaled(
                        self._main_page.size(),
                        Qt.IgnoreAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )

            self._survey_background_label.setGeometry(self._survey_page.rect())
            if not self._survey_background_pixmap.isNull():
                self._survey_background_label.setPixmap(
                    self._survey_background_pixmap.scaled(
                        self._survey_page.size(),
                        Qt.IgnoreAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )

            main_logo_x, main_logo_y, main_logo_width, main_logo_height = self._scale_rect(self._main_logo_rect)
            self._main_logo_button.setGeometry(main_logo_x, main_logo_y, main_logo_width, main_logo_height)

            self._analysis_label.setGeometry(self._analysis_page.rect())
            if self._analysis_frame is not None:
                self._set_analysis_frame(self._analysis_frame)

            scale = min(
                self.width() / max(1.0, float(self._base_size[0])),
                self.height() / max(1.0, float(self._base_size[1])),
            )
            title_font = QFont("Malgun Gothic", max(14, int(round(24 * scale))))
            title_font.setWeight(QFont.DemiBold)
            option_font = QFont("Malgun Gothic", max(11, int(round(15 * scale))))
            option_font.setWeight(QFont.Medium)
            button_font = QFont("Malgun Gothic", max(10, int(round(15 * scale))))
            button_font.setWeight(QFont.DemiBold)

            for label, rect in self._titles:
                x, y, width, height = self._scale_rect(rect)
                label.setGeometry(x, y, width, height)
                label.setFont(title_font)

            for index, (checkbox, rect) in enumerate(self._checkboxes):
                x, y, width, height = self._scale_rect(rect)
                checkbox.setGeometry(x, y, width, height)
                checkbox.setFont(option_font)
                checkbox.setChecked(self._option_states[index])

            button_x, button_y, button_width, button_height = self._scale_rect(self._survey_button_rect)
            self._complete_button.setGeometry(button_x, button_y, button_width, button_height)
            self._complete_button.setFont(button_font)

        def _cleanup_analysis(self) -> None:
            if self._frame_timer.isActive():
                self._frame_timer.stop()
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            if self._backend is not None:
                self._backend.close()
                self._backend = None

            if self._smoother is not None:
                self._summary_text = summarize(self._smoother.latest())

        def resizeEvent(self, event) -> None:
            self._apply_layout()
            super().resizeEvent(event)

        def showEvent(self, event) -> None:
            super().showEvent(event)
            self._apply_layout()

        def closeEvent(self, event) -> None:
            self._cleanup_analysis()
            super().closeEvent(event)

        def keyPressEvent(self, event) -> None:
            if event.key() in (Qt.Key_Escape, Qt.Key_Q):
                self.reject()
                return
            if event.key() == Qt.Key_C and event.modifiers() & Qt.ControlModifier:
                self.reject()
                return
            if self._current_page == "main" and event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
                self._start_intro_transition()
                return
            if self._current_page == "analysis":
                if event.key() == Qt.Key_R and self._smoother is not None:
                    self._smoother.reset()
                    self._last_tongue_preview = None
                    self._tongue_preview_hold_frames = 0
                    return
                if event.key() == Qt.Key_F:
                    self._flip_view = not self._flip_view
                    return
                if event.key() == Qt.Key_O:
                    self._rotation = next_rotation(self._rotation)
                    return
            super().keyPressEvent(event)


def run_qt_bodycheck(
    args: argparse.Namespace,
    main_layout: np.ndarray,
    survey_layout: np.ndarray,
    typea_layout: np.ndarray,
) -> int:
    if QApplication is None or QDialog is None or QPixmap is None or Qt is None:
        return 1

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = SurveyDialog(
        MAIN_LAYOUT_PATH,
        compute_main_logo_rect(main_layout),
        QNA_LAYOUT_PATH,
        compute_survey_button_rect(survey_layout),
        typea_layout,
        args,
        initial_states=[False] * SURVEY_OPTION_COUNT,
    )
    screen = app.primaryScreen()
    if screen is not None:
        dialog.setGeometry(screen.geometry())
    dialog.showFullScreen()
    keepalive_timer = QTimer()
    keepalive_timer.setInterval(150)
    keepalive_timer.timeout.connect(lambda: None)
    keepalive_timer.start()

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(_signum, _frame) -> None:
        if dialog.isVisible():
            dialog.reject()
        else:
            app.quit()

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        dialog.exec_()
    finally:
        keepalive_timer.stop()
        signal.signal(signal.SIGINT, previous_sigint_handler)
    summary = dialog.final_summary()
    if summary:
        print(summary)
    return 0


def point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
    rect_x, rect_y, rect_width, rect_height = rect
    return rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height


def find_survey_option_index(x: int, y: int) -> Optional[int]:
    option_index = 0
    for section in SURVEY_SECTION_SPECS:
        for option_text, click_rect in zip(section.options, section.click_rects):
            if option_text and point_in_rect(x, y, click_rect):
                return option_index
            option_index += 1
    return None


def draw_survey_complete_button(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    hovered: bool,
    accent_color: Color,
) -> None:
    x, y, width, height = rect
    radius = max(18, int(height * SURVEY_BUTTON_RADIUS_RATIO))
    font_size = max(18, int(height * 0.38)) + (1 if hovered else 0)

    shadow_rect = (x, y + 6, width, height)
    shadow_color = blend_colors(accent_color, (205, 210, 220), 0.58)
    blend_rounded_panel(
        frame,
        shadow_rect,
        fill_color=shadow_color,
        alpha=0.22,
        radius=radius,
    )

    fill_color = blend_colors(accent_color, (255, 255, 255), 0.12 if hovered else 0.2)
    border_color = blend_colors(accent_color, (108, 120, 150), 0.38 if hovered else 0.48)
    blend_rounded_panel(
        frame,
        rect,
        fill_color=fill_color,
        alpha=0.97,
        radius=radius,
        border_color=border_color,
        border_thickness=3 if hovered else 2,
    )

    highlight_rect = (x + 10, y + 8, width - 20, max(18, int(height * 0.32)))
    blend_rounded_panel(
        frame,
        highlight_rect,
        fill_color=(255, 255, 255),
        alpha=0.18 if hovered else 0.11,
        radius=max(12, radius - 8),
    )

    text_color = SURVEY_TEXT_COLOR
    draw_centered_korean_text(frame, rect, SURVEY_BUTTON_TEXT, font_size, text_color)


def draw_text_in_rect_on_pil(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    text: str,
    font_size: int,
    color: Color,
    align: str = "left",
    padding_x: int = 0,
    render_scale: int = 1,
) -> None:
    if not text:
        return

    scaled_font_size = max(1, int(font_size * max(1, render_scale)))
    font = get_survey_korean_font(scaled_font_size)
    if font is None:
        return

    scale = max(1, int(render_scale))
    x, y, width, height = rect
    x *= scale
    y *= scale
    width *= scale
    height *= scale
    scaled_padding_x = padding_x * scale
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    if align == "center":
        text_x = x + max(0.0, (width - text_width) / 2.0)
    else:
        text_x = x + scaled_padding_x
    text_y = y + max(0.0, (height - text_height) / 2.0) - float(scale)
    draw.text(
        (int(round(text_x)), int(round(text_y))),
        text,
        font=font,
        fill=(color[2], color[1], color[0], 255),
    )


def render_survey_base_frame(background: np.ndarray) -> np.ndarray:
    canvas = background.copy()
    if Image is None or ImageDraw is None:
        return canvas

    pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
    render_scale = max(1, SURVEY_TEXT_RENDER_SCALE)
    text_layer = Image.new(
        "RGBA",
        (pil_image.width * render_scale, pil_image.height * render_scale),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(text_layer)
    title_color = SURVEY_TEXT_COLOR
    option_color = SURVEY_TEXT_COLOR

    for section in SURVEY_SECTION_SPECS:
        draw_text_in_rect_on_pil(
            draw,
            section.title_rect,
            section.title,
            SURVEY_TITLE_FONT_SIZE,
            title_color,
            align="center",
            render_scale=render_scale,
        )
        for option_text, option_rect in zip(section.options, section.option_rects):
            draw_text_in_rect_on_pil(
                draw,
                option_rect,
                option_text,
                SURVEY_OPTION_FONT_SIZE,
                option_color,
                align="left",
                padding_x=18,
                render_scale=render_scale,
            )

    if render_scale > 1:
        lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        text_layer = text_layer.resize(pil_image.size, lanczos)

    pil_image.alpha_composite(text_layer)
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def draw_survey_checkboxes(frame: np.ndarray, ui_state: UiState) -> None:
    option_index = 0
    checked_fill = (177, 198, 244)
    checked_border = (122, 151, 226)
    check_color = SURVEY_TEXT_COLOR
    hover_fill = (240, 244, 252)

    for section in SURVEY_SECTION_SPECS:
        for checkbox_rect, option_text in zip(section.checkbox_rects, section.options):
            if not option_text:
                option_index += 1
                continue

            x, y, width, height = checkbox_rect
            if ui_state.survey_hovered_option == option_index and not ui_state.survey_option_states[option_index]:
                blend_rounded_panel(
                    frame,
                    (x + 2, y + 2, max(1, width - 4), max(1, height - 4)),
                    fill_color=hover_fill,
                    alpha=0.85,
                    radius=5,
                )

            if ui_state.survey_option_states[option_index]:
                blend_rounded_panel(
                    frame,
                    (x + 2, y + 2, max(1, width - 4), max(1, height - 4)),
                    fill_color=checked_fill,
                    alpha=0.95,
                    radius=5,
                    border_color=checked_border,
                    border_thickness=1,
                )
                check_points = np.array(
                    [
                        (x + 5, y + (height // 2)),
                        (x + 9, y + height - 6),
                        (x + width - 5, y + 5),
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(frame, [check_points], False, check_color, 2, cv2.LINE_AA)

            option_index += 1


def compose_survey_page(background: np.ndarray, ui_state: UiState) -> np.ndarray:
    canvas = ui_state.survey_base_frame.copy() if ui_state.survey_base_frame is not None else background.copy()
    draw_survey_checkboxes(canvas, ui_state)
    draw_survey_complete_button(
        canvas,
        ui_state.survey_button_rect,
        ui_state.survey_button_hovered,
        ui_state.survey_button_accent,
    )
    return canvas


def handle_mouse_event(event: int, x: int, y: int, flags: int, ui_state: UiState) -> None:
    del flags
    if ui_state is None or ui_state.current_page != "survey":
        return

    set_window_arrow_cursor(WINDOW_NAME)
    button_hovered = point_in_rect(x, y, ui_state.survey_button_rect)
    ui_state.survey_button_hovered = button_hovered
    ui_state.survey_hovered_option = find_survey_option_index(x, y)

    if event == cv2.EVENT_LBUTTONUP and button_hovered:
        ui_state.survey_completed = True
        return

    if event == cv2.EVENT_LBUTTONUP and ui_state.survey_hovered_option is not None:
        option_index = ui_state.survey_hovered_option
        ui_state.survey_option_states[option_index] = not ui_state.survey_option_states[option_index]


def is_window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def inset_rect(rect: Tuple[int, int, int, int], margin: int) -> Tuple[int, int, int, int]:
    x, y, width, height = rect
    return x + margin, y + margin, max(1, width - (margin * 2)), max(1, height - (margin * 2))


def resolve_ui_target(
    layout: np.ndarray,
    search_rect: Tuple[int, int, int, int],
    fallback_radius: int,
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    x, y, width, height = search_rect
    margin = 40
    search_x = max(0, x - margin)
    search_y = max(0, y - margin)
    search_w = min(layout.shape[1] - search_x, width + (margin * 2))
    search_h = min(layout.shape[0] - search_y, height + (margin * 2))
    roi = layout[search_y : search_y + search_h, search_x : search_x + search_w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # The inner camera box border is a pale blue line; detect it and fill its contour.
    blue_mask = cv2.inRange(hsv, (90, 20, 150), (130, 255, 255))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask)

    best_idx = -1
    best_score = None
    for idx in range(1, component_count):
        comp_x, comp_y, comp_w, comp_h, area = stats[idx]
        if area < 1000:
            continue

        global_x = search_x + comp_x
        global_y = search_y + comp_y
        score = (
            abs(global_x - x)
            + abs(global_y - y)
            + abs(comp_w - width)
            + abs(comp_h - height)
        )
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        rect = inset_rect(search_rect, BODY_CAMERA_MARGIN)
        return rect, rounded_rect_mask(rect[2], rect[3], fallback_radius)

    comp_x, comp_y, comp_w, comp_h, _ = stats[best_idx]
    component = np.where(labels == best_idx, 255, 0).astype(np.uint8)
    local_component = component[comp_y : comp_y + comp_h, comp_x : comp_x + comp_w]
    contours, _ = cv2.findContours(local_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        rect = inset_rect(search_rect, BODY_CAMERA_MARGIN)
        return rect, rounded_rect_mask(rect[2], rect[3], fallback_radius)

    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros((comp_h, comp_w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1, cv2.LINE_AA)
    return (search_x + comp_x, search_y + comp_y, comp_w, comp_h), mask


def resolve_body_camera_target(
    layout: np.ndarray,
    search_rect: Tuple[int, int, int, int] = BODY_CAMERA_RECT,
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    return resolve_ui_target(layout, search_rect, BODY_CAMERA_RADIUS)


def resolve_face_camera_target(
    layout: np.ndarray,
    search_rect: Tuple[int, int, int, int],
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    return resolve_ui_target(layout, search_rect, FACE_CAMERA_RADIUS)


def rounded_rect_mask(width: int, height: int, radius: int) -> np.ndarray:
    radius = max(1, min(radius, width // 2, height // 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1, cv2.LINE_AA)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1, cv2.LINE_AA)

    corners = (
        (radius, radius),
        (width - radius - 1, radius),
        (radius, height - radius - 1),
        (width - radius - 1, height - radius - 1),
    )
    for center in corners:
        cv2.circle(mask, center, radius, 255, -1, cv2.LINE_AA)
    return mask


def paste_frame_in_mask(
    canvas: np.ndarray,
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    mask: np.ndarray,
    fit_mode: str = "cover",
) -> None:
    if frame is None:
        return

    x, y, width, height = rect
    background_roi = canvas[y : y + height, x : x + width]
    if fit_mode == "contain":
        fitted = fit_frame_to_canvas_contain(frame, width, height, background_roi)
    else:
        fitted = fit_frame_to_canvas(frame, width, height)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]

    roi = canvas[y : y + height, x : x + width].astype(np.float32)
    fitted_float = fitted.astype(np.float32)
    blended = (fitted_float * alpha) + (roi * (1.0 - alpha))
    canvas[y : y + height, x : x + width] = blended.astype(np.uint8)


@lru_cache(maxsize=8)
def get_korean_font(font_size: int):
    if ImageFont is None:
        return None

    candidates = (
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), font_size)
    return ImageFont.load_default()


@lru_cache(maxsize=8)
def get_survey_korean_font(font_size: int):
    if ImageFont is None:
        return None

    candidates = (
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/Hancom Gothic Regular.ttf"),
        Path("C:/Windows/Fonts/Hancom Gothic Bold.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), font_size)
    return get_korean_font(font_size)


def draw_centered_korean_text(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    text: str,
    font_size: int,
    color: Color,
) -> None:
    x, y, width, height = rect
    if Image is None or ImageDraw is None:
        fallback = "Detecting..."
        text_size, _ = cv2.getTextSize(fallback, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        text_x = x + max(0, (width - text_size[0]) // 2)
        text_y = y + max(text_size[1] + 6, (height + text_size[1]) // 2)
        cv2.putText(frame, fallback, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return

    roi = frame[y : y + height, x : x + width]
    pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = get_korean_font(font_size)
    if font is None:
        return
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (width - text_width) / 2.0
    text_y = (height - text_height) / 2.0 - 2.0
    draw.text((text_x, text_y), text, font=font, fill=(color[2], color[1], color[0]))
    frame[y : y + height, x : x + width] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def blend_rounded_panel(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    fill_color: Color,
    alpha: float,
    radius: int,
    border_color: Optional[Color] = None,
    border_thickness: int = 2,
) -> None:
    x, y, width, height = rect
    overlay = frame[y : y + height, x : x + width].copy()
    fill = np.full((height, width, 3), fill_color, dtype=np.uint8)
    mask = rounded_rect_mask(width, height, radius)
    fill_alpha = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    blended = (fill.astype(np.float32) * fill_alpha) + (overlay.astype(np.float32) * (1.0 - fill_alpha))
    frame[y : y + height, x : x + width] = blended.astype(np.uint8)

    if border_color is not None and border_thickness > 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            shifted = [contour + np.array([[[x, y]]]) for contour in contours]
            cv2.drawContours(frame, shifted, -1, border_color, border_thickness, cv2.LINE_AA)


def draw_preview_wait_badge(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    message: str,
) -> None:
    x, y, width, height = rect
    badge_width = min(width - 36, 228)
    badge_height = 56
    badge_x = x + (width - badge_width) // 2
    badge_y = y + (height - badge_height) // 2
    badge_rect = (badge_x, badge_y, badge_width, badge_height)

    blend_rounded_panel(
        frame,
        badge_rect,
        fill_color=(34, 44, 60),
        alpha=0.82,
        radius=18,
        border_color=(189, 207, 239),
        border_thickness=2,
    )
    cv2.circle(frame, (badge_x + 24, badge_y + (badge_height // 2)), 6, (216, 232, 255), -1, cv2.LINE_AA)
    text_rect = (badge_x + 38, badge_y, badge_width - 44, badge_height)
    draw_centered_korean_text(frame, text_rect, message, 21, OVERLAY_TEXT_PRIMARY)


def create_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(str(cascade_path))


def detect_primary_face(frame: np.ndarray, face_detector: cv2.CascadeClassifier) -> Optional[Tuple[int, int, int, int]]:
    if frame is None or face_detector.empty():
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(120, 120),
    )
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(width), int(height)


def expand_face_rect(face_rect: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    x, y, width, height = face_rect
    frame_height, frame_width = frame_shape[:2]
    pad_x = int(width * 0.42)
    pad_top = int(height * 0.40)
    pad_bottom = int(height * 0.56)
    left = max(0, x - pad_x)
    top = max(0, y - pad_top)
    right = min(frame_width, x + width + pad_x)
    bottom = min(frame_height, y + height + pad_bottom)
    return left, top, max(1, right - left), max(1, bottom - top)


def adjust_rect_to_aspect(
    rect: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
    target_aspect_ratio: float,
    vertical_bias: float = 0.0,
) -> Tuple[int, int, int, int]:
    x, y, width, height = rect
    frame_height, frame_width = frame_shape[:2]
    if target_aspect_ratio <= 0.0:
        return rect

    center_x = x + (width / 2.0)
    center_y = y + (height / 2.0) + (height * vertical_bias)
    new_width = float(width)
    new_height = float(height)
    current_ratio = new_width / max(new_height, 1.0)
    if current_ratio < target_aspect_ratio:
        new_width = new_height * target_aspect_ratio
    else:
        new_height = new_width / target_aspect_ratio

    new_width = min(float(frame_width), new_width)
    new_height = min(float(frame_height), new_height)
    left = int(round(center_x - (new_width / 2.0)))
    top = int(round(center_y - (new_height / 2.0)))
    right = int(round(left + new_width))
    bottom = int(round(top + new_height))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > frame_width:
        left -= right - frame_width
        right = frame_width
    if bottom > frame_height:
        top -= bottom - frame_height
        bottom = frame_height

    left = max(0, left)
    top = max(0, top)
    right = min(frame_width, max(left + 1, right))
    bottom = min(frame_height, max(top + 1, bottom))
    return left, top, max(1, right - left), max(1, bottom - top)


def crop_rect(frame: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = rect
    return frame[y : y + height, x : x + width].copy()


def crop_face_region(
    frame: np.ndarray,
    face_detector: cv2.CascadeClassifier,
    target_aspect_ratio: Optional[float] = None,
) -> Optional[np.ndarray]:
    face_rect = detect_primary_face(frame, face_detector)
    if face_rect is None:
        return None

    crop_rect_xywh = expand_face_rect(face_rect, frame.shape)
    if target_aspect_ratio is not None:
        crop_rect_xywh = adjust_rect_to_aspect(
            crop_rect_xywh,
            frame.shape,
            target_aspect_ratio,
            vertical_bias=0.05,
        )
    return crop_rect(frame, crop_rect_xywh)


def extract_mouth_region(
    face_crop: Optional[np.ndarray],
    target_aspect_ratio: Optional[float] = None,
) -> Optional[np.ndarray]:
    if face_crop is None:
        return None

    face_height, face_width = face_crop.shape[:2]
    mouth_rect = (
        int(face_width * 0.18),
        int(face_height * 0.52),
        max(1, int(face_width * 0.64)),
        max(1, int(face_height * 0.42)),
    )
    if target_aspect_ratio is not None:
        mouth_rect = adjust_rect_to_aspect(
            mouth_rect,
            face_crop.shape,
            target_aspect_ratio,
            vertical_bias=0.18,
        )
    return crop_rect(face_crop, mouth_rect)


def detect_tongue_in_face(
    face_crop: Optional[np.ndarray],
    preview_aspect_ratio: Optional[float] = None,
) -> Tuple[bool, float, Optional[np.ndarray]]:
    if face_crop is None:
        return False, 0.0, None

    mouth = extract_mouth_region(face_crop)
    preview = extract_mouth_region(face_crop, preview_aspect_ratio)
    if mouth.size == 0:
        return False, 0.0, preview

    hsv = cv2.cvtColor(mouth, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(mouth, cv2.COLOR_BGR2LAB)
    red_mask_1 = cv2.inRange(hsv, (0, 24, 35), (24, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (160, 24, 35), (179, 255, 255))
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    warm_mask = cv2.inRange(lab[:, :, 1], 138, 255)

    blue, green, red = cv2.split(mouth)
    red_dominant = (
        (red.astype(np.int16) >= green.astype(np.int16) + 10)
        & (red.astype(np.int16) >= blue.astype(np.int16) + 8)
        & (red >= 55)
    )
    bright_enough = hsv[:, :, 2] >= 40
    mask = np.where(
        (red_mask > 0) & (warm_mask > 0) & red_dominant & bright_enough,
        255,
        0,
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, preview

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    mouth_area = float(mouth.shape[0] * mouth.shape[1])
    if mouth_area <= 0.0:
        return False, 0.0, preview

    x, y, width, height = cv2.boundingRect(contour)
    area_ratio = area / mouth_area
    center_x_ratio = (x + (width * 0.5)) / max(1.0, mouth.shape[1])
    lower_bias = (y + (height * 0.5)) / max(1.0, mouth.shape[0])
    lower_half = mask[int(mouth.shape[0] * 0.52) :, :]
    upper_half = mask[: int(mouth.shape[0] * 0.44), :]
    center_band = mask[:, int(mouth.shape[1] * 0.22) : int(mouth.shape[1] * 0.78)]
    lower_ratio = cv2.countNonZero(lower_half) / max(1.0, lower_half.size)
    upper_ratio = cv2.countNonZero(upper_half) / max(1.0, upper_half.size)
    center_ratio = cv2.countNonZero(center_band) / max(1.0, center_band.size)
    center_bias = max(0.0, 1.0 - abs(center_x_ratio - 0.5) * 2.0)
    score = (
        area_ratio * 1.4
        + lower_ratio * 1.1
        + center_ratio * 0.8
        + lower_bias * 0.35
        + center_bias * 0.25
        - upper_ratio * 0.2
    )
    detected = (
        area_ratio >= 0.008
        and lower_bias >= 0.45
        and width >= mouth.shape[1] * 0.10
        and height >= mouth.shape[0] * 0.10
        and center_bias >= 0.22
        and lower_ratio >= 0.012
        and center_ratio >= 0.010
        and lower_ratio >= upper_ratio * 0.85
    )
    return detected, score, preview


def compose_ui_frame(
    background: np.ndarray,
    body_preview_frame: np.ndarray,
    face_preview_frame: Optional[np.ndarray],
    tongue_preview_frame: Optional[np.ndarray],
    readings: Optional[Dict[str, MetricReading]],
    guidance: str,
    detection_ok: bool,
    body_target: Tuple[Tuple[int, int, int, int], np.ndarray],
    face_target: Tuple[Tuple[int, int, int, int], np.ndarray],
    tongue_target: Tuple[Tuple[int, int, int, int], np.ndarray],
) -> np.ndarray:
    canvas = background.copy()
    body_rect, body_mask = body_target
    face_rect, face_mask = face_target
    tongue_rect, tongue_mask = tongue_target

    paste_frame_in_mask(canvas, body_preview_frame, body_rect, body_mask, fit_mode="cover")
    if face_preview_frame is not None:
        paste_frame_in_mask(canvas, face_preview_frame, face_rect, face_mask, fit_mode="cover")
    else:
        draw_preview_wait_badge(canvas, face_rect, "얼굴 감지중...")

    if tongue_preview_frame is not None:
        paste_frame_in_mask(canvas, tongue_preview_frame, tongue_rect, tongue_mask, fit_mode="cover")
    else:
        draw_preview_wait_badge(canvas, tongue_rect, "혀 감지중...")
    draw_overlay(canvas, body_rect, readings, guidance, detection_ok)
    return canvas


def compose_loading_ui_frame(
    background: np.ndarray,
    guidance: str,
    body_target: Tuple[Tuple[int, int, int, int], np.ndarray],
    face_target: Tuple[Tuple[int, int, int, int], np.ndarray],
    tongue_target: Tuple[Tuple[int, int, int, int], np.ndarray],
) -> np.ndarray:
    canvas = background.copy()
    body_rect, _ = body_target
    face_rect, _ = face_target
    tongue_rect, _ = tongue_target

    draw_preview_wait_badge(canvas, body_rect, "준비중...")
    draw_preview_wait_badge(canvas, face_rect, "준비중...")
    draw_preview_wait_badge(canvas, tongue_rect, "준비중...")
    draw_overlay(canvas, body_rect, None, guidance, False)
    return canvas


class MetricSmoother:
    def __init__(self, keys: Iterable[str], size: int) -> None:
        self._values: Dict[str, Deque[float]] = {key: deque(maxlen=size) for key in keys}

    def reset(self) -> None:
        for queue in self._values.values():
            queue.clear()

    def update(self, raw_metrics: Dict[str, float]) -> Dict[str, float]:
        smoothed: Dict[str, float] = {}
        for key, value in raw_metrics.items():
            self._values[key].append(value)
            smoothed[key] = float(np.mean(self._values[key]))
        return smoothed

    def latest(self) -> Dict[str, Optional[float]]:
        latest_values: Dict[str, Optional[float]] = {}
        for key, queue in self._values.items():
            latest_values[key] = float(np.mean(queue)) if queue else None
        return latest_values


class BodyPostureAnalyzer:
    def __init__(self, landmark_enum, min_visibility: float = 0.55) -> None:
        self.min_visibility = min_visibility
        self._pose_landmarks = landmark_enum

    def _get_point(self, landmarks, landmark_id: int) -> Point:
        landmark = landmarks[int(landmark_id)]
        return Point(
            x=float(landmark.x),
            y=float(landmark.y),
            visibility=float(getattr(landmark, "visibility", 1.0) or 0.0),
        )

    def _required_points(self, landmarks) -> Dict[str, Point]:
        ids = self._pose_landmarks
        return {
            "nose": self._get_point(landmarks, ids.NOSE),
            "left_shoulder": self._get_point(landmarks, ids.LEFT_SHOULDER),
            "right_shoulder": self._get_point(landmarks, ids.RIGHT_SHOULDER),
            "left_hip": self._get_point(landmarks, ids.LEFT_HIP),
            "right_hip": self._get_point(landmarks, ids.RIGHT_HIP),
            "left_knee": self._get_point(landmarks, ids.LEFT_KNEE),
            "right_knee": self._get_point(landmarks, ids.RIGHT_KNEE),
            "left_ankle": self._get_point(landmarks, ids.LEFT_ANKLE),
            "right_ankle": self._get_point(landmarks, ids.RIGHT_ANKLE),
        }

    def _is_in_frame(self, point: Point) -> bool:
        return 0.02 <= point.x <= 0.98 and 0.02 <= point.y <= 0.98

    def _validate_front_standing_pose(self, points: Dict[str, Point]) -> Optional[str]:
        for name, point in points.items():
            if point.visibility < self.min_visibility:
                return f"Low landmark confidence: {name}"
            if name != "nose" and not self._is_in_frame(point):
                return "Move back so your whole body fits inside the frame."

        shoulder_width = distance(points["left_shoulder"], points["right_shoulder"])
        hip_width = distance(points["left_hip"], points["right_hip"])
        body_height = distance(points["nose"], midpoint(points["left_ankle"], points["right_ankle"]))
        if shoulder_width < 0.08 or hip_width < 0.07 or body_height < 0.45:
            return "Stand farther from the camera and face the front."
        return None

    def analyze(self, landmarks) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        if not landmarks:
            return None, "No body detected. Step back until your full body is visible."

        points = self._required_points(landmarks)
        validation_message = self._validate_front_standing_pose(points)
        if validation_message:
            return None, validation_message

        shoulder_center = midpoint(points["left_shoulder"], points["right_shoulder"])
        hip_center = midpoint(points["left_hip"], points["right_hip"])
        shoulder_width = max(distance(points["left_shoulder"], points["right_shoulder"]), 1e-6)

        metrics = {
            "shoulder_tilt": horizontal_angle_deg(points["left_shoulder"], points["right_shoulder"]),
            "pelvis_tilt": horizontal_angle_deg(points["left_hip"], points["right_hip"]),
            "torso_lean": vertical_lean_deg(shoulder_center, hip_center),
            "center_shift": abs(shoulder_center.x - hip_center.x) / shoulder_width,
        }
        return metrics, None


class SolutionsPoseBackend:
    def __init__(self) -> None:
        pose_module = mp.solutions.pose
        self.backend_name = "solutions"
        self.landmark_enum = pose_module.PoseLandmark
        self.connections = pose_module.POSE_CONNECTIONS
        self._drawer = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._pose = pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> PoseDetection:
        del timestamp_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        rgb.flags.writeable = True
        landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
        return PoseDetection(landmarks=landmarks, raw_result=result)

    def draw(self, frame: np.ndarray, detection: PoseDetection) -> None:
        if detection.raw_result and detection.raw_result.pose_landmarks:
            self._drawer.draw_landmarks(
                frame,
                detection.raw_result.pose_landmarks,
                self.connections,
                landmark_drawing_spec=self._styles.get_default_pose_landmarks_style(),
            )

    def close(self) -> None:
        self._pose.close()


class TasksPoseBackend:
    def __init__(self, model_variant: str, model_path: Optional[str]) -> None:
        self.backend_name = f"tasks:{model_variant}"
        self._model_path = resolve_task_model_path(model_variant, model_path)
        self.landmark_enum = mp.tasks.vision.PoseLandmark
        self.connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        self._drawer = mp.tasks.vision.drawing_utils
        self._styles = mp.tasks.vision.drawing_styles

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            output_segmentation_masks=False,
        )
        self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> PoseDetection:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._pose.detect_for_video(mp_image, timestamp_ms)
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        return PoseDetection(landmarks=landmarks, raw_result=result)

    def draw(self, frame: np.ndarray, detection: PoseDetection) -> None:
        if detection.landmarks:
            self._drawer.draw_landmarks(
                frame,
                detection.landmarks,
                self.connections,
                landmark_drawing_spec=self._styles.get_default_pose_landmarks_style(),
            )

    def close(self) -> None:
        self._pose.close()


def resolve_task_model_path(model_variant: str, model_path: Optional[str]) -> Path:
    if model_variant not in MODEL_URLS:
        raise ValueError(f"Unsupported model variant: {model_variant}")

    if model_path:
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Pose model file not found: {resolved}")
        return resolved

    target = Path(__file__).resolve().parent / "models" / "mediapipe" / f"pose_landmarker_{model_variant}.task"
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe pose model ({model_variant}) to {target} ...")
    try:
        urllib.request.urlretrieve(MODEL_URLS[model_variant], target)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Failed to download the MediaPipe pose model.\n"
            "You can try again later or pass --model-path with a local .task file."
        ) from exc
    return target


def create_pose_backend(model_variant: str, model_path: Optional[str]):
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return SolutionsPoseBackend()
    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and hasattr(mp.tasks.vision, "PoseLandmarker"):
        return TasksPoseBackend(model_variant=model_variant, model_path=model_path)
    raise RuntimeError(
        "The installed mediapipe package does not provide a usable pose API.\n"
        "Install a build with MediaPipe Solutions or Tasks support."
    )


def draw_text_block(
    frame: np.ndarray,
    origin: Tuple[int, int],
    text: str,
    color: Color = (255, 255, 255),
    scale: float = 0.65,
    thickness: int = 2,
) -> int:
    x, y = origin
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return y + int(28 * scale + 14)


def wrap_text(text: str, max_width: int, scale: float, thickness: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]

    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        width, _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        if width <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def draw_overlay(
    frame: np.ndarray,
    preview_rect: Tuple[int, int, int, int],
    readings: Optional[Dict[str, MetricReading]],
    guidance: str,
    detection_ok: bool,
) -> None:
    x, y, width, height = preview_rect
    panel_w = min(width - 24, 420)
    panel_x = x + 12
    panel_y = y + 12
    guidance_lines = wrap_text(guidance, panel_w - 28, 0.54, 2)
    panel_h = 68 + (26 * len(guidance_lines)) + (24 * len(readings) if readings else 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text_y = draw_text_block(frame, (panel_x + 14, panel_y + 28), "BodyCheck", OVERLAY_TEXT_PRIMARY, 0.72, 2)
    text_y = draw_text_block(
        frame,
        (panel_x + 14, text_y),
        "Q: quit | R: reset | F: flip | O: rotate",
        OVERLAY_TEXT_SECONDARY,
        0.52,
        1,
    )

    status_color = OVERLAY_TEXT_GUIDE_READY if detection_ok else OVERLAY_TEXT_GUIDE_WAIT
    for line in guidance_lines:
        text_y = draw_text_block(frame, (panel_x + 14, text_y + 4), line, status_color, 0.54, 2)

    if not readings:
        return

    text_y += 4
    for reading in readings.values():
        display_value = (
            f"{reading.value:.1f} {reading.unit}"
            if reading.unit != "ratio"
            else f"{reading.value:.3f}"
        )
        line = f"{reading.label}: {display_value}"
        metric_text_color = OVERLAY_TEXT_METRIC_COLORS.get(reading.status, OVERLAY_TEXT_PRIMARY)
        text_y = draw_text_block(frame, (panel_x + 14, text_y), line, metric_text_color, 0.55, 2)


def draw_reference_lines(frame: np.ndarray, landmarks, landmark_enum) -> None:
    if not landmarks:
        return

    height, width = frame.shape[:2]

    def pixel_point(landmark_id: int) -> Tuple[int, int]:
        landmark = landmarks[int(landmark_id)]
        return int(float(landmark.x) * width), int(float(landmark.y) * height)

    left_shoulder = pixel_point(landmark_enum.LEFT_SHOULDER)
    right_shoulder = pixel_point(landmark_enum.RIGHT_SHOULDER)
    left_hip = pixel_point(landmark_enum.LEFT_HIP)
    right_hip = pixel_point(landmark_enum.RIGHT_HIP)
    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) // 2,
        (left_shoulder[1] + right_shoulder[1]) // 2,
    )
    hip_center = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)

    cv2.line(frame, left_shoulder, right_shoulder, (255, 215, 0), 2, cv2.LINE_AA)
    cv2.line(frame, left_hip, right_hip, (255, 140, 0), 2, cv2.LINE_AA)
    cv2.line(frame, shoulder_center, hip_center, (60, 210, 255), 2, cv2.LINE_AA)
    cv2.line(
        frame,
        (hip_center[0], 0),
        (hip_center[0], frame.shape[0]),
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )


def summarize(readings: Dict[str, Optional[float]]) -> str:
    available = {k: v for k, v in readings.items() if v is not None}
    if not available:
        return "No stable posture summary collected."

    classified = [classify_metric(name, value) for name, value in available.items()]
    worst_status = "good"
    for item in classified:
        if item.status == "bad":
            worst_status = "bad"
            break
        if item.status == "warn":
            worst_status = "warn"

    if worst_status == "good":
        overall = "Overall: balanced standing posture."
    elif worst_status == "warn":
        overall = "Overall: mild asymmetry detected. Recheck after relaxing your stance."
    else:
        overall = "Overall: clear asymmetry detected. Consider repeated checks or professional evaluation."

    metric_text = ", ".join(
        f"{item.label}={item.value:.1f}{item.unit if item.unit != 'ratio' else ''}"
        if item.unit != "ratio"
        else f"{item.label}={item.value:.3f}"
        for item in classified
    )
    return f"{overall}\n{metric_text}"


def open_camera(camera_index: int, width: int, height: int) -> Tuple[cv2.VideoCapture, Tuple[int, int]]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened() and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {camera_index}.")

    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    actual_size = configure_capture_mode(cap, width, height)
    return cap, actual_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time standing posture check with webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to open. Default: 0")
    parser.add_argument("--width", type=int, default=720, help="Requested capture width. Default: 720")
    parser.add_argument("--height", type=int, default=1280, help="Requested capture height. Default: 1280")
    parser.add_argument(
        "--smooth",
        type=int,
        default=12,
        help="Number of frames used for moving average. Default: 12",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.55,
        help="Minimum landmark visibility to trust posture values. Default: 0.55",
    )
    parser.add_argument(
        "--model-variant",
        choices=tuple(MODEL_URLS.keys()),
        default="lite",
        help="Pose model size for Tasks API. Default: lite",
    )
    parser.add_argument(
        "--model-path",
        help="Optional local path to a MediaPipe pose .task model file.",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable mirror view.",
    )
    parser.add_argument(
        "--rotation",
        choices=tuple(ROTATION_FLAGS.keys()),
        default="none",
        help="Rotate the camera feed for portrait display. Default: none",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    flip_view = not args.no_flip
    rotation = args.rotation
    try:
        main_layout = load_ui_layout(MAIN_LAYOUT_PATH)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    if main_layout is None:
        print(f"UI layout image not found: {MAIN_LAYOUT_PATH}")
        return 1

    try:
        survey_layout = load_ui_layout(QNA_LAYOUT_PATH)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    if survey_layout is None:
        print(f"UI layout image not found: {QNA_LAYOUT_PATH}")
        return 1

    if QApplication is not None and QDialog is not None and QPixmap is not None and Qt is not None:
        try:
            typea_layout = load_ui_layout(TYPEA_LAYOUT_PATH)
        except RuntimeError as exc:
            print(str(exc))
            return 1

        if typea_layout is None:
            print(f"UI layout image not found: {TYPEA_LAYOUT_PATH}")
            return 1
        return run_qt_bodycheck(args, main_layout, survey_layout, typea_layout)

    window_initialized = False
    survey_height, survey_width = survey_layout.shape[:2]
    ui_state = UiState(
        survey_button_rect=compute_survey_button_rect(survey_layout),
        survey_button_accent=estimate_layout_accent_color(survey_layout),
        survey_option_states=[False] * SURVEY_OPTION_COUNT,
        survey_base_frame=render_survey_base_frame(survey_layout),
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse_event, ui_state)
    set_window_fullscreen_borderless(WINDOW_NAME)
    set_window_arrow_cursor(WINDOW_NAME)
    print("Starting BodyCheck. Press Q to quit. Complete the survey page to continue.")
    while not ui_state.survey_completed:
        if is_window_closed(WINDOW_NAME):
            return 0

        survey_frame = compose_survey_page(survey_layout, ui_state)
        cv2.imshow(WINDOW_NAME, survey_frame)
        set_window_arrow_cursor(WINDOW_NAME)

        key = cv2.waitKey(16) & 0xFF
        if is_window_closed(WINDOW_NAME):
            return 0
        if key == 3:
            return 0
        if key == ord("q"):
            return 0
        if key in (13, 32):
            ui_state.survey_completed = True
    window_initialized = True

    smoother: Optional[MetricSmoother] = None
    backend = None
    cap = None
    try:
        try:
            typea_layout = load_ui_layout(TYPEA_LAYOUT_PATH)
        except RuntimeError as exc:
            print(str(exc))
            return 1

        if typea_layout is None:
            print(f"UI layout image not found: {TYPEA_LAYOUT_PATH}")
            return 1

        if not window_initialized:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        set_window_fullscreen_borderless(WINDOW_NAME)
        set_window_arrow_cursor(WINDOW_NAME)

        smoother = MetricSmoother(METRIC_SPECS.keys(), max(3, args.smooth))
        display_height, display_width = typea_layout.shape[:2]
        ui_body_target = resolve_body_camera_target(typea_layout)
        ui_face_target = resolve_face_camera_target(typea_layout, FACE_A_CAMERA_RECT)
        ui_tongue_target = resolve_face_camera_target(typea_layout, FACE_B_CAMERA_RECT)
        face_preview_aspect_ratio = ui_face_target[0][2] / max(1.0, ui_face_target[0][3])
        tongue_preview_aspect_ratio = ui_tongue_target[0][2] / max(1.0, ui_tongue_target[0][3])
        set_window_fullscreen_borderless(WINDOW_NAME)

        loading_frame = compose_loading_ui_frame(
            typea_layout,
            "--",
            ui_body_target,
            ui_face_target,
            ui_tongue_target,
        )
        cv2.imshow(WINDOW_NAME, loading_frame)
        cv2.waitKey(1)

        try:
            backend = create_pose_backend(model_variant=args.model_variant, model_path=args.model_path)
        except Exception as exc:
            print(str(exc))
            return 1

        loading_frame = compose_loading_ui_frame(
            typea_layout,
            "--",
            ui_body_target,
            ui_face_target,
            ui_tongue_target,
        )
        cv2.imshow(WINDOW_NAME, loading_frame)
        cv2.waitKey(1)

        analyzer = BodyPostureAnalyzer(backend.landmark_enum, min_visibility=args.min_visibility)
        face_detector = create_face_detector()
        last_tongue_preview: Optional[np.ndarray] = None
        tongue_preview_hold_frames = 0

        try:
            cap, capture_size = open_camera(args.camera, args.width, args.height)
        except RuntimeError as exc:
            print(str(exc))
            return 1

        print(
            f"Survey complete. Starting posture analysis with {backend.backend_name}. "
            f"Camera capture: {capture_size[0]}x{capture_size[1]}"
        )

        while True:
            if is_window_closed(WINDOW_NAME):
                break

            ok, frame = cap.read()
            if not ok:
                print("Failed to read from webcam.")
                break

            frame = apply_rotation(frame, rotation)
            if flip_view:
                frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()

            timestamp_ms = int(time.monotonic() * 1000)
            detection = backend.detect(frame, timestamp_ms)

            readings: Optional[Dict[str, MetricReading]] = None
            guidance = "Stand naturally facing the camera with your full body visible."
            detection_ok = False

            metrics, message = analyzer.analyze(detection.landmarks)
            if metrics is not None:
                detection_ok = True
                averaged = smoother.update(metrics)
                readings = {
                    metric_name: classify_metric(metric_name, metric_value)
                    for metric_name, metric_value in averaged.items()
                }
                guidance = "Stable reading. Relax your arms and keep both feet on the floor."
            else:
                guidance = message or guidance

            face_preview = crop_face_region(
                raw_frame,
                face_detector,
                target_aspect_ratio=face_preview_aspect_ratio,
            )
            tongue_preview = None
            if face_preview is not None:
                tongue_detected, _tongue_score, mouth_preview = detect_tongue_in_face(
                    face_preview,
                    preview_aspect_ratio=tongue_preview_aspect_ratio,
                )
                if tongue_detected and mouth_preview is not None:
                    last_tongue_preview = mouth_preview.copy()
                    tongue_preview = mouth_preview
                    tongue_preview_hold_frames = TONGUE_PREVIEW_HOLD_FRAMES
                elif last_tongue_preview is not None and tongue_preview_hold_frames > 0:
                    tongue_preview = last_tongue_preview
                    tongue_preview_hold_frames -= 1
                else:
                    last_tongue_preview = None
                    tongue_preview_hold_frames = 0
            else:
                last_tongue_preview = None
                tongue_preview_hold_frames = 0

            if detection.landmarks:
                backend.draw(frame, detection)
                draw_reference_lines(frame, detection.landmarks, backend.landmark_enum)

            display_frame = compose_ui_frame(
                typea_layout,
                frame,
                face_preview,
                tongue_preview,
                readings,
                guidance,
                detection_ok,
                ui_body_target,
                ui_face_target,
                ui_tongue_target,
            )
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if is_window_closed(WINDOW_NAME):
                break
            if key == 3:
                break
            if key == ord("q"):
                break
            if key == ord("r"):
                smoother.reset()
                last_tongue_preview = None
                tongue_preview_hold_frames = 0
                print("Averaged posture metrics and tongue preview reset.")
            if key == ord("f"):
                flip_view = not flip_view
            if key == ord("o"):
                rotation = next_rotation(rotation)
    finally:
        if cap is not None:
            cap.release()
        if backend is not None:
            backend.close()
        cv2.destroyAllWindows()

    if smoother is not None:
        print(summarize(smoother.latest()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        raise SystemExit(0)
