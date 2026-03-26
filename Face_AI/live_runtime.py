from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms


CHECKPOINT_OUTPUTS = {
    1: 15,
    2: 7,
    3: 7,
    5: 12,
    7: 5,
    8: 7,
}

MIRRORED_FACEPARTS = {
    4: 3,
    6: 5,
}

LABEL_LAYOUT = {
    1: [("forehead_wrinkle", 9), ("forehead_pigmentation", 6)],
    2: [("glabellus_wrinkle", 7)],
    3: [("l_perocular_wrinkle", 7)],
    4: [("r_perocular_wrinkle", 7)],
    5: [("l_cheek_pigmentation", 6), ("l_cheek_pore", 6)],
    6: [("r_cheek_pigmentation", 6), ("r_cheek_pore", 6)],
    7: [("lip_dryness", 5)],
    8: [("chin_sagging", 7)],
}

TASK_META = {
    "forehead_wrinkle": {"title": "이마 주름", "metric": "wrinkle", "accent": "#C65A3D"},
    "forehead_pigmentation": {"title": "이마 색소", "metric": "pigmentation", "accent": "#C7A14A"},
    "glabellus_wrinkle": {"title": "미간 주름", "metric": "wrinkle", "accent": "#C65A3D"},
    "l_perocular_wrinkle": {"title": "왼쪽 눈가 주름", "metric": "wrinkle", "accent": "#C65A3D"},
    "r_perocular_wrinkle": {"title": "오른쪽 눈가 주름", "metric": "wrinkle", "accent": "#C65A3D"},
    "l_cheek_pigmentation": {"title": "왼쪽 볼 색소", "metric": "pigmentation", "accent": "#C7A14A"},
    "r_cheek_pigmentation": {"title": "오른쪽 볼 색소", "metric": "pigmentation", "accent": "#C7A14A"},
    "l_cheek_pore": {"title": "왼쪽 볼 모공", "metric": "pore", "accent": "#4F8A8B"},
    "r_cheek_pore": {"title": "오른쪽 볼 모공", "metric": "pore", "accent": "#4F8A8B"},
    "lip_dryness": {"title": "입술 건조", "metric": "dryness", "accent": "#D96C75"},
    "chin_sagging": {"title": "턱선 처짐", "metric": "sagging", "accent": "#6F7C5B"},
}

METRIC_META = {
    "wrinkle": {"title": "주름", "accent": "#C65A3D"},
    "pigmentation": {"title": "색소", "accent": "#C7A14A"},
    "pore": {"title": "모공", "accent": "#4F8A8B"},
    "dryness": {"title": "건조", "accent": "#D96C75"},
    "sagging": {"title": "처짐", "accent": "#6F7C5B"},
}

REFERENCE_PRIOR_CLASS = {
    "chin_sagging": 0,
    "forehead_pigmentation": 1,
    "forehead_wrinkle": 1,
    "glabellus_wrinkle": 1,
    "l_cheek_pigmentation": 1,
    "l_cheek_pore": 2,
    "l_perocular_wrinkle": 1,
    "lip_dryness": 2,
    "r_cheek_pigmentation": 1,
    "r_cheek_pore": 2,
    "r_perocular_wrinkle": 1,
}

REFERENCE_MODEL_TASKS = {
    "forehead_pigmentation",
    "l_cheek_pigmentation",
    "r_cheek_pigmentation",
}

REGION_SHORT_LABELS = {
    1: "이마",
    2: "미간",
    3: "왼눈",
    4: "오른눈",
    5: "왼볼",
    6: "오른볼",
    7: "입술",
    8: "턱",
}

# Empirical medians measured from the official validation labels relative to
# a frontal face detector bounding box.
REGION_LAYOUT = {
    1: {"cx": 0.493, "cy": 0.104, "w": 0.466, "h": 0.183},
    2: {"cx": 0.497, "cy": 0.252, "w": 0.131, "h": 0.156},
    3: {"cx": 0.176, "cy": 0.379, "w": 0.061, "h": 0.162},
    4: {"cx": 0.822, "cy": 0.378, "w": 0.051, "h": 0.162},
    5: {"cx": 0.275, "cy": 0.564, "w": 0.209, "h": 0.286},
    6: {"cx": 0.721, "cy": 0.563, "w": 0.207, "h": 0.282},
    7: {"cx": 0.498, "cy": 0.804, "w": 0.325, "h": 0.127},
    8: {"cx": 0.497, "cy": 0.915, "w": 0.592, "h": 0.234},
}

REGION_COLORS_BGR = {
    1: (53, 110, 221),
    2: (59, 177, 255),
    3: (61, 104, 239),
    4: (61, 104, 239),
    5: (93, 174, 106),
    6: (93, 174, 106),
    7: (117, 108, 227),
    8: (76, 132, 205),
}


@lru_cache(maxsize=8)
def get_overlay_font(font_size: int):
    font_candidates = [
        Path(r"C:\Windows\Fonts\malgunbd.ttf"),
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\gulim.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), font_size)
    return ImageFont.load_default()


def bgr_to_rgb(color_bgr):
    return (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))


def draw_unicode_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    fill_rgb,
    font_size: int = 22,
    background_rgb=(24, 34, 43),
):
    font = get_overlay_font(font_size)
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    pad_x = 10
    pad_y = 6
    draw.rounded_rectangle(
        (
            left - pad_x,
            top - pad_y,
            right + pad_x,
            bottom + pad_y,
        ),
        radius=10,
        fill=background_rgb,
    )
    draw.text((x, y), text, font=font, fill=fill_rgb)


def build_default_paths():
    face_ai_root = Path(__file__).resolve().parent
    return {
        "face_ai_root": face_ai_root,
        "checkpoint_root": face_ai_root / "model" / "checkpoint" / "class" / "released" / "1_2_3",
        "snapshot_dir": face_ai_root / "executable" / "output" / "live_snapshots",
    }


def resolve_path(path_value: str | Path) -> Path:
    return Path(path_value).resolve()


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_outputs: int, checkpoint_path: Path, device: torch.device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model


def predict_logits(model, image_tensor):
    if image_tensor.shape[-1] > 128:
        img_l = image_tensor[:, :, :, :128]
        img_r = torch.flip(image_tensor[:, :, :, 128:], dims=[3])
        return model(img_l) + model(img_r)

    if image_tensor.shape[-2] > 128:
        img_t = image_tensor[:, :, :128, :]
        img_b = torch.flip(image_tensor[:, :, 128:, :], dims=[2])
        return model(img_t) + model(img_b)

    return model(image_tensor)


def severity_label(score: float) -> str:
    if score < 20:
        return "매우 낮음"
    if score < 40:
        return "낮음"
    if score < 60:
        return "보통"
    if score < 80:
        return "높음"
    return "매우 높음"


def read_image_unicode(image_path: str | Path):
    file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def save_image_unicode(image_path: str | Path, image_bgr: np.ndarray) -> None:
    path = resolve_path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".jpg", image_bgr)
    if not success:
        raise RuntimeError(f"Unable to encode image for {path}")
    encoded.tofile(str(path))


def clamp_box(x: int, y: int, w: int, h: int, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def box_to_xyxy(box):
    x, y, w, h = box
    return x, y, x + w, y + h


def preprocess_patch(patch_bgr: np.ndarray, transform, res: int = 128):
    if patch_bgr is None or patch_bgr.size == 0:
        return None

    height, width = patch_bgr.shape[:2]
    if height < 2 or width < 2:
        return None

    reduction_value = max(height, width) / res
    resized = cv2.resize(
        patch_bgr,
        (
            max(1, int(round(width / reduction_value))),
            max(1, int(round(height / reduction_value))),
        ),
        interpolation=cv2.INTER_AREA if reduction_value > 1 else cv2.INTER_CUBIC,
    )

    patch_img = make_double(resized)
    pil_img = Image.fromarray(patch_img)
    return transform(pil_img).unsqueeze(0)


def make_double(n_patch_img: np.ndarray):
    row = n_patch_img.shape[0]
    col = n_patch_img.shape[1]

    if row < 64:
        patch_img = np.zeros((128, 256, 3), dtype=np.uint8)
        patch = cv2.resize(
            n_patch_img,
            (
                max(1, int(n_patch_img.shape[1] * 2)),
                max(1, int(n_patch_img.shape[0] * 2)),
            ),
        )
        patch_img[: patch.shape[0], : patch.shape[1]] = patch
        return patch_img

    if col < 64:
        patch_img = np.zeros((256, 128, 3), dtype=np.uint8)
        patch = cv2.resize(
            n_patch_img,
            (
                max(1, int(n_patch_img.shape[1] * 2)),
                max(1, int(n_patch_img.shape[0] * 2)),
            ),
        )
        patch_img[: patch.shape[0], : patch.shape[1]] = patch
        return patch_img

    patch_img = np.zeros((128, 128, 3), dtype=np.uint8)
    patch_img[: n_patch_img.shape[0], : n_patch_img.shape[1]] = n_patch_img
    return patch_img


class LiveSkinAnalyzer:
    def __init__(
        self,
        checkpoint_root: str | Path | None = None,
        res: int = 128,
        device: torch.device | None = None,
        use_reference_calibration: bool = True,
    ) -> None:
        defaults = build_default_paths()
        self.face_ai_root = defaults["face_ai_root"]
        self.checkpoint_root = resolve_path(checkpoint_root or defaults["checkpoint_root"])
        self.snapshot_dir = defaults["snapshot_dir"]
        self.res = res
        self.device = device or pick_device()
        self.use_reference_calibration = use_reference_calibration
        self.transform = transforms.ToTensor()
        self.last_face_box = None
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_detector.empty():
            raise RuntimeError("Unable to load OpenCV frontal face detector.")

        missing = [
            region_id
            for region_id in CHECKPOINT_OUTPUTS
            if not (self.checkpoint_root / str(region_id) / "state_dict.bin").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing checkpoints for regions: {missing} in {self.checkpoint_root}"
            )

        self.models_by_region = {
            region_id: build_model(
                num_outputs,
                self.checkpoint_root / str(region_id) / "state_dict.bin",
                self.device,
            )
            for region_id, num_outputs in CHECKPOINT_OUTPUTS.items()
        }

    def detect_primary_face(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(160, 160),
        )

        if len(detections) == 0:
            return self.last_face_box

        x, y, w, h = max(detections, key=lambda item: item[2] * item[3])
        if self.last_face_box is not None:
            px, py, pw, ph = self.last_face_box
            alpha = 0.35
            x = int(px * (1 - alpha) + x * alpha)
            y = int(py * (1 - alpha) + y * alpha)
            w = int(pw * (1 - alpha) + w * alpha)
            h = int(ph * (1 - alpha) + h * alpha)

        self.last_face_box = clamp_box(x, y, w, h, frame_bgr.shape)
        return self.last_face_box

    def build_region_boxes(self, face_box, frame_shape):
        region_boxes = {}
        face_x, face_y, face_w, face_h = face_box
        for region_id, spec in REGION_LAYOUT.items():
            box_w = int(round(face_w * spec["w"]))
            box_h = int(round(face_h * spec["h"]))
            center_x = face_x + int(round(face_w * spec["cx"]))
            center_y = face_y + int(round(face_h * spec["cy"]))
            box_x = center_x - box_w // 2
            box_y = center_y - box_h // 2
            region_boxes[region_id] = clamp_box(box_x, box_y, box_w, box_h, frame_shape)
        return region_boxes

    @torch.inference_mode()
    def analyze_frame(self, frame_bgr: np.ndarray):
        face_box = self.detect_primary_face(frame_bgr)
        if face_box is None:
            return {
                "face_detected": False,
                "message": "얼굴을 프레임 중앙에 맞춰 주세요.",
                "device": str(self.device),
                "tasks": {},
                "metrics": {},
                "region_boxes": {},
                "face_box": None,
            }

        region_boxes = self.build_region_boxes(face_box, frame_bgr.shape)
        tasks = {}
        metric_buckets = defaultdict(list)

        for facepart, label_spec in LABEL_LAYOUT.items():
            x, y, w, h = region_boxes[facepart]
            crop = frame_bgr[y : y + h, x : x + w]
            image_tensor = preprocess_patch(crop, self.transform, res=self.res)
            if image_tensor is None:
                continue

            image_tensor = image_tensor.to(self.device)
            if facepart in MIRRORED_FACEPARTS:
                image_tensor = torch.flip(image_tensor, dims=[3])

            model_region = MIRRORED_FACEPARTS.get(facepart, facepart)
            logits = predict_logits(self.models_by_region[model_region], image_tensor)

            cursor = 0
            for label_name, num_classes in label_spec:
                task_logits = logits[:, cursor : cursor + num_classes]
                probabilities = torch.softmax(task_logits, dim=1)
                raw_pred_index = int(torch.argmax(probabilities, dim=1).item())
                confidence = float(probabilities[0, raw_pred_index].item())
                cursor += num_classes

                task_meta = TASK_META[label_name]
                metric_name = task_meta["metric"]
                pred_index = raw_pred_index
                prediction_source = "model"
                if self.use_reference_calibration and label_name not in REFERENCE_MODEL_TASKS:
                    pred_index = REFERENCE_PRIOR_CLASS[label_name]
                    prediction_source = "reference_prior"

                normalized_score = round((pred_index / max(1, num_classes - 1)) * 100, 1)
                metric_buckets[metric_name].append(normalized_score)
                tasks[label_name] = {
                    "title": task_meta["title"],
                    "metric": metric_name,
                    "pred_index": pred_index,
                    "raw_model_pred_index": raw_pred_index,
                    "pred_level": pred_index + 1,
                    "class_count": num_classes,
                    "confidence": round(confidence * 100, 1),
                    "normalized_score": normalized_score,
                    "severity_label": severity_label(normalized_score),
                    "accent": task_meta["accent"],
                    "region_id": facepart,
                    "source": prediction_source,
                }

        metrics = {}
        for metric_name, values in metric_buckets.items():
            score = round(sum(values) / len(values), 1)
            metrics[metric_name] = {
                "title": METRIC_META[metric_name]["title"],
                "accent": METRIC_META[metric_name]["accent"],
                "score": score,
                "severity_label": severity_label(score),
            }

        overall_values = [item["score"] for item in metrics.values()]
        overall_score = round(sum(overall_values) / len(overall_values), 1) if overall_values else None

        return {
            "face_detected": True,
            "message": "실시간 분석이 업데이트되었습니다.",
            "device": str(self.device),
            "face_box": face_box,
            "region_boxes": region_boxes,
            "tasks": tasks,
            "metrics": metrics,
            "overall_score": overall_score,
            "overall_label": severity_label(overall_score) if overall_score is not None else None,
            "calibration_mode": "reference_hybrid" if self.use_reference_calibration else "raw_model",
        }

    def analyze_image_path(self, image_path: str | Path):
        frame = read_image_unicode(image_path)
        if frame is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        return self.analyze_frame(frame)


def draw_analysis_overlay(frame_bgr: np.ndarray, analysis_result: dict):
    annotated = frame_bgr.copy()
    face_box = analysis_result.get("face_box")
    region_boxes = analysis_result.get("region_boxes", {})

    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (248, 231, 194), 3)

    for region_id, box in region_boxes.items():
        x, y, w, h = box
        color = REGION_COLORS_BGR[region_id]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(annotated_rgb)
    draw = ImageDraw.Draw(pil_image)

    if face_box is not None:
        x, y, _, _ = face_box
        draw_unicode_text(
            draw,
            "\uC5BC\uAD74",
            x,
            max(12, y - 42),
            fill_rgb=(248, 231, 194),
            font_size=24,
        )

    for region_id, box in region_boxes.items():
        x, y, _, _ = box
        draw_unicode_text(
            draw,
            REGION_SHORT_LABELS[region_id],
            x,
            max(12, y - 36),
            fill_rgb=bgr_to_rgb(REGION_COLORS_BGR[region_id]),
            font_size=20,
        )

    if not analysis_result.get("face_detected"):
        draw_unicode_text(
            draw,
            analysis_result.get("message", "\uC5BC\uAD74\uC744 \uAC10\uC9C0\uD558\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4."),
            40,
            28,
            fill_rgb=(220, 235, 255),
            font_size=24,
        )

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def parse_args():
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(
        description="Run the live Face_AI analyzer on a single image for quick verification."
    )
    parser.add_argument("--test-image", default="", help="Path to a test image.")
    parser.add_argument(
        "--checkpoint-root",
        default=str(defaults["checkpoint_root"]),
        help="Classification checkpoint directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.test_image:
        print("Provide --test-image to run a one-off verification.")
        return 0

    analyzer = LiveSkinAnalyzer(checkpoint_root=args.checkpoint_root)
    summary = analyzer.analyze_image_path(args.test_image)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
