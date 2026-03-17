from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
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
UI_LAYOUT_PATH = Path(__file__).resolve().parent / "UI" / "TypeA.png"
# Approximate search box for the inner rounded camera area in UI/TypeA.png.
BODY_CAMERA_RECT = (102, 141, 485, 818)
FACE_A_CAMERA_RECT = (676, 139, 549, 325)
FACE_B_CAMERA_RECT = (1296, 139, 549, 325)
BODY_CAMERA_MARGIN = 0
BODY_CAMERA_RADIUS = 23
FACE_CAMERA_RADIUS = 26


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


def load_ui_layout() -> Optional[np.ndarray]:
    if not UI_LAYOUT_PATH.exists():
        return None

    layout = cv2.imread(str(UI_LAYOUT_PATH))
    if layout is None:
        raise RuntimeError(f"Failed to load UI layout image: {UI_LAYOUT_PATH}")
    return layout


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
        fill_color=(241, 247, 255),
        alpha=0.92,
        radius=18,
        border_color=(163, 191, 236),
        border_thickness=2,
    )
    cv2.circle(frame, (badge_x + 24, badge_y + (badge_height // 2)), 6, (118, 198, 255), -1, cv2.LINE_AA)
    text_rect = (badge_x + 38, badge_y, badge_width - 44, badge_height)
    draw_centered_korean_text(frame, text_rect, message, 21, (67, 97, 156))


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


def crop_face_region(frame: np.ndarray, face_detector: cv2.CascadeClassifier) -> Optional[np.ndarray]:
    face_rect = detect_primary_face(frame, face_detector)
    if face_rect is None:
        return None

    x, y, width, height = expand_face_rect(face_rect, frame.shape)
    return frame[y : y + height, x : x + width].copy()


def detect_tongue_in_face(face_crop: Optional[np.ndarray]) -> Tuple[bool, float]:
    if face_crop is None:
        return False, 0.0

    face_height, face_width = face_crop.shape[:2]
    mouth_x1 = int(face_width * 0.22)
    mouth_x2 = int(face_width * 0.78)
    mouth_y1 = int(face_height * 0.56)
    mouth_y2 = int(face_height * 0.94)
    mouth = face_crop[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
    if mouth.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(mouth, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(mouth, cv2.COLOR_BGR2LAB)
    red_mask_1 = cv2.inRange(hsv, (0, 35, 45), (18, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (160, 35, 45), (179, 255, 255))
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    warm_mask = cv2.inRange(lab[:, :, 1], 148, 255)
    mask = cv2.bitwise_and(red_mask, warm_mask)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    mouth_area = float(mouth.shape[0] * mouth.shape[1])
    if mouth_area <= 0.0:
        return False, 0.0

    x, y, width, height = cv2.boundingRect(contour)
    area_ratio = area / mouth_area
    lower_bias = (y + (height * 0.5)) / max(1.0, mouth.shape[0])
    score = area_ratio * (0.5 + lower_bias)
    detected = area_ratio >= 0.018 and lower_bias >= 0.42 and width >= mouth.shape[1] * 0.12
    return detected, score


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
        paste_frame_in_mask(canvas, face_preview_frame, face_rect, face_mask, fit_mode="contain")
    else:
        draw_preview_wait_badge(canvas, face_rect, "얼굴 감지중...")

    if tongue_preview_frame is not None:
        paste_frame_in_mask(canvas, tongue_preview_frame, tongue_rect, tongue_mask, fit_mode="contain")
    else:
        draw_preview_wait_badge(canvas, tongue_rect, "혀 감지중...")
    draw_overlay(canvas, body_rect, readings, guidance, detection_ok)
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

    text_y = draw_text_block(frame, (panel_x + 14, panel_y + 28), "BodyCheck", (255, 255, 255), 0.72, 2)
    text_y = draw_text_block(
        frame,
        (panel_x + 14, text_y),
        "Q: quit | R: reset | F: flip | O: rotate",
        (210, 210, 210),
        0.52,
        1,
    )

    status_color = (80, 220, 120) if detection_ok else (0, 180, 255)
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
        text_y = draw_text_block(frame, (panel_x + 14, text_y), line, reading.color, 0.55, 2)


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
    smoother = MetricSmoother(METRIC_SPECS.keys(), max(3, args.smooth))
    flip_view = not args.no_flip
    rotation = args.rotation
    try:
        ui_layout = load_ui_layout()
    except RuntimeError as exc:
        print(str(exc))
        return 1

    ui_body_target: Optional[Tuple[Tuple[int, int, int, int], np.ndarray]] = None
    ui_face_target: Optional[Tuple[Tuple[int, int, int, int], np.ndarray]] = None
    ui_tongue_target: Optional[Tuple[Tuple[int, int, int, int], np.ndarray]] = None
    if ui_layout is not None:
        display_height, display_width = ui_layout.shape[:2]
        ui_body_target = resolve_body_camera_target(ui_layout)
        ui_face_target = resolve_face_camera_target(ui_layout, FACE_A_CAMERA_RECT)
        ui_tongue_target = resolve_face_camera_target(ui_layout, FACE_B_CAMERA_RECT)
    else:
        display_width, display_height = portrait_display_size(args.width, args.height)

    try:
        backend = create_pose_backend(model_variant=args.model_variant, model_path=args.model_path)
    except Exception as exc:
        print(str(exc))
        return 1

    analyzer = BodyPostureAnalyzer(backend.landmark_enum, min_visibility=args.min_visibility)
    face_detector = create_face_detector()
    last_tongue_preview: Optional[np.ndarray] = None

    try:
        cap, capture_size = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        backend.close()
        print(str(exc))
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_width, display_height)
    print(
        f"Starting BodyCheck. Press Q to quit. Using {backend.backend_name}. "
        f"Camera capture: {capture_size[0]}x{capture_size[1]}"
    )
    try:
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

            face_preview = crop_face_region(raw_frame, face_detector)
            tongue_preview = last_tongue_preview
            if face_preview is not None and last_tongue_preview is None:
                tongue_detected, _tongue_score = detect_tongue_in_face(face_preview)
                if tongue_detected:
                    last_tongue_preview = face_preview.copy()
                    tongue_preview = last_tongue_preview

            if detection.landmarks:
                backend.draw(frame, detection)
                draw_reference_lines(frame, detection.landmarks, backend.landmark_enum)

            if ui_layout is not None:
                display_frame = compose_ui_frame(
                    ui_layout,
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
            else:
                display_frame = fit_frame_to_canvas(frame, display_width, display_height)
                draw_overlay(
                    display_frame,
                    (12, 12, display_width - 24, display_height - 24),
                    readings,
                    guidance,
                    detection_ok,
                )
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if is_window_closed(WINDOW_NAME):
                break
            if key == ord("q"):
                break
            if key == ord("r"):
                smoother.reset()
                last_tongue_preview = None
                print("Averaged posture metrics and tongue preview reset.")
            if key == ord("f"):
                flip_view = not flip_view
            if key == ord("o"):
                rotation = next_rotation(rotation)
    finally:
        cap.release()
        backend.close()
        cv2.destroyAllWindows()

    print(summarize(smoother.latest()))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
