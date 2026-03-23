import argparse
import asyncio
import json
import logging
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - depends on local environment
    mp = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("fastapi_camera_backend")

VIDEO_CLOCK_RATE = 90000
DEFAULT_CAPTURE_WIDTH = 3840
DEFAULT_CAPTURE_HEIGHT = 2160
DEFAULT_CAPTURE_FPS = 30
MJPEG_QUALITY = 95
RESOLUTION_TOLERANCE = 0.9
BODY_STREAM_MAX_WIDTH = 1920
BODY_STREAM_MAX_HEIGHT = 1080
BODY_STREAM_MAX_FPS = 12
BODY_DETECTION_MAX_DIMENSION = 1280
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
COMMON_RESOLUTIONS = [
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1600, 1200),
    (1280, 720),
    (1024, 768),
    (800, 600),
    (640, 480),
]


@dataclass(slots=True)
class ServerSettings:
    host: str
    port: int
    camera_indices: list[int]
    width: int
    height: int
    fps: int


@dataclass(slots=True)
class CaptureHandle:
    capture: cv2.VideoCapture
    backend_name: str
    codec_name: str
    width: int
    height: int
    fps: float


def resolve_pose_model_path() -> Path:
    bundled_candidates = [
        Path(__file__).resolve().parents[2] / "old" / "models" / "mediapipe" / "pose_landmarker_lite.task",
        Path(__file__).resolve().parent / "models" / "mediapipe" / "pose_landmarker_lite.task",
    ]
    for candidate in bundled_candidates:
        if candidate.exists():
            return candidate

    target = bundled_candidates[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading MediaPipe pose model to %s", target)
    urllib.request.urlretrieve(POSE_MODEL_URL, target)
    return target


class BodySkeletonRenderer:
    def __init__(self) -> None:
        if mp is None:
            raise RuntimeError("mediapipe is required for the body skeleton stream.")

        self._backend_type = ""
        self._timestamp_ms = 0

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            pose_module = mp.solutions.pose
            drawing_utils = mp.solutions.drawing_utils
            self._backend_type = "solutions"
            self._pose = pose_module.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self._drawing = drawing_utils
            self._connections = pose_module.POSE_CONNECTIONS
        elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and hasattr(mp.tasks.vision, "PoseLandmarker"):
            self._backend_type = "tasks"
            self._drawing = mp.tasks.vision.drawing_utils
            self._connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(resolve_pose_model_path())),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.55,
                min_pose_presence_confidence=0.55,
                min_tracking_confidence=0.55,
                output_segmentation_masks=False,
            )
            self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        else:
            raise RuntimeError("The installed mediapipe package does not expose a usable pose API.")

        self._landmark_spec = self._drawing.DrawingSpec(
            color=(191, 167, 20),
            thickness=2,
            circle_radius=3,
        )
        self._connection_spec = self._drawing.DrawingSpec(
            color=(236, 201, 112),
            thickness=4,
            circle_radius=2,
        )
        self._background_cache_key: tuple[int, int] | None = None
        self._background_cache: np.ndarray | None = None

    def _build_background(self, width: int, height: int) -> np.ndarray:
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, height, dtype=np.float32),
            np.linspace(0.0, 1.0, width, dtype=np.float32),
            indexing="ij",
        )
        blue = 255 - (20.0 * yy) - (6.0 * np.abs(xx - 0.5))
        green = 250 - (24.0 * yy)
        red = 246 - (12.0 * yy)
        background = np.stack([blue, green, red], axis=-1).clip(0, 255).astype(np.uint8)

        glow = background.copy()
        center = (width // 2, int(height * 0.86))
        cv2.ellipse(
            glow,
            center,
            (max(60, int(width * 0.23)), max(20, int(height * 0.04))),
            0,
            0,
            360,
            (245, 229, 188),
            2,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            glow,
            center,
            (max(46, int(width * 0.16)), max(12, int(height * 0.025))),
            0,
            0,
            360,
            (255, 243, 214),
            1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(glow, 0.36, background, 0.64, 0, background)
        return background

    def _background(self, width: int, height: int) -> np.ndarray:
        cache_key = (width, height)
        if self._background_cache_key != cache_key or self._background_cache is None:
            self._background_cache_key = cache_key
            self._background_cache = self._build_background(width, height)
        return self._background_cache

    def render(self, frame: np.ndarray, detection_frame: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        source_frame = frame
        analysis_frame = detection_frame if detection_frame is not None else frame
        canvas = source_frame.copy()
        rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        canvas = cv2.convertScaleAbs(canvas, alpha=1.02, beta=4)

        if self._backend_type == "solutions":
            rgb.flags.writeable = False
            result = self._pose.process(rgb)
            rgb.flags.writeable = True
            landmarks = result.pose_landmarks
        else:
            self._timestamp_ms += 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = self._pose.detect_for_video(mp_image, self._timestamp_ms)
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

        if landmarks:
            self._drawing.draw_landmarks(
                canvas,
                landmarks,
                self._connections,
                landmark_drawing_spec=self._landmark_spec,
                connection_drawing_spec=self._connection_spec,
            )
            return canvas, True

        return canvas, False

    def close(self) -> None:
        self._pose.close()


class OfferPayload(BaseModel):
    sdp: str
    type: str
    requestedTracks: int | None = None


def candidate_backends() -> list[tuple[str, int]]:
    if sys.platform.startswith("win"):
        return [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("ANY", cv2.CAP_ANY),
        ]
    return [("ANY", cv2.CAP_ANY)]


def candidate_codecs() -> list[tuple[str, int | None]]:
    codecs: list[tuple[str, int | None]] = [("MJPG", cv2.VideoWriter_fourcc(*"MJPG"))]
    if sys.platform.startswith("win"):
        codecs.append(("YUY2", cv2.VideoWriter_fourcc(*"YUY2")))
    codecs.append(("DEFAULT", None))
    return codecs


def build_resolution_candidates(width: int, height: int) -> list[tuple[int, int]]:
    requested = (max(1, width), max(1, height))
    requested_pixels = requested[0] * requested[1]
    candidates = [requested]
    for resolution in COMMON_RESOLUTIONS:
        if resolution == requested:
            continue
        if resolution[0] * resolution[1] <= requested_pixels:
            candidates.append(resolution)
    return candidates


def create_capture(index: int, backend: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, backend)
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def configure_capture(
    capture: cv2.VideoCapture,
    width: int,
    height: int,
    fps: int,
    fourcc: int | None,
) -> None:
    if fourcc is not None:
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    time.sleep(0.15)


def resolution_matches(actual_width: int, actual_height: int, target_width: int, target_height: int) -> bool:
    return (
        actual_width >= int(target_width * RESOLUTION_TOLERANCE)
        and actual_height >= int(target_height * RESOLUTION_TOLERANCE)
    )


def reopen_capture(
    index: int,
    backend_name: str,
    backend: int,
    codec_name: str,
    width: int,
    height: int,
    fps: int,
) -> CaptureHandle:
    capture = create_capture(index, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Camera index {index} could not be reopened with backend {backend_name}.")

    codec_fourcc = None
    for candidate_name, candidate_fourcc in candidate_codecs():
        if candidate_name == codec_name:
            codec_fourcc = candidate_fourcc
            break

    configure_capture(capture, width, height, fps, codec_fourcc)
    success, _ = capture.read()
    if not success:
        capture.release()
        raise RuntimeError(f"Camera index {index} failed while reopening {width}x{height}.")

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps)
    LOGGER.info(
        "Camera %s reopened with backend %s, codec %s at %sx%s @ %.2ffps",
        index,
        backend_name,
        codec_name,
        actual_width,
        actual_height,
        actual_fps,
    )
    return CaptureHandle(
        capture=capture,
        backend_name=backend_name,
        codec_name=codec_name,
        width=actual_width,
        height=actual_height,
        fps=actual_fps,
    )


def open_capture(index: int, width: int, height: int, fps: int) -> CaptureHandle:
    best_match: tuple[str, int, str, int, int, float] | None = None
    candidates = build_resolution_candidates(width, height)

    for backend_name, backend in candidate_backends():
        capture = create_capture(index, backend)
        if not capture.isOpened():
            capture.release()
            continue

        for codec_name, codec_fourcc in candidate_codecs():
            for target_width, target_height in candidates:
                configure_capture(capture, target_width, target_height, fps, codec_fourcc)
                success, _ = capture.read()
                if not success:
                    continue

                actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or target_width)
                actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or target_height)
                actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps)
                LOGGER.info(
                    "Camera %s probe via %s/%s requested %sx%s @ %sfps -> actual %sx%s @ %.2ffps",
                    index,
                    backend_name,
                    codec_name,
                    target_width,
                    target_height,
                    fps,
                    actual_width,
                    actual_height,
                    actual_fps,
                )

                if resolution_matches(actual_width, actual_height, target_width, target_height):
                    LOGGER.info(
                        "Camera %s opened with backend %s, codec %s at %sx%s @ %.2ffps",
                        index,
                        backend_name,
                        codec_name,
                        actual_width,
                        actual_height,
                        actual_fps,
                    )
                    return CaptureHandle(
                        capture=capture,
                        backend_name=backend_name,
                        codec_name=codec_name,
                        width=actual_width,
                        height=actual_height,
                        fps=actual_fps,
                    )

                if best_match is None or (actual_width * actual_height) > (best_match[3] * best_match[4]):
                    best_match = (
                        backend_name,
                        backend,
                        codec_name,
                        actual_width,
                        actual_height,
                        actual_fps,
                    )

        capture.release()

    if best_match is not None:
        backend_name, backend, codec_name, actual_width, actual_height, actual_fps = best_match
        LOGGER.warning(
            "Camera %s did not reach requested %sx%s. Falling back to best detected %sx%s @ %.2ffps via %s/%s.",
            index,
            width,
            height,
            actual_width,
            actual_height,
            actual_fps,
            backend_name,
            codec_name,
        )
        return reopen_capture(
            index=index,
            backend_name=backend_name,
            backend=backend,
            codec_name=codec_name,
            width=actual_width,
            height=actual_height,
            fps=max(1, int(round(actual_fps or fps))),
        )

    raise RuntimeError(f"Camera index {index} could not be opened by any backend.")


def probe_cameras(max_index: int) -> list[int]:
    available: list[int] = []
    for index in range(max_index):
        for _, backend in candidate_backends():
            capture = cv2.VideoCapture(index, backend)
            try:
                if capture.isOpened():
                    success, _ = capture.read()
                    if success:
                        available.append(index)
                        break
            finally:
                capture.release()
    return available


class OpenCVCameraTrack(VideoStreamTrack):
    def __init__(self, camera_index: int, width: int, height: int, fps: int) -> None:
        super().__init__()
        self.camera_index = camera_index
        self._capture_handle = open_capture(camera_index, width, height, fps)
        self.fps = max(1.0, self._capture_handle.fps or float(fps))
        self._capture = self._capture_handle.capture
        self._timestamp = 0
        self._start: float | None = None
        LOGGER.info(
            "Track camera %s streaming at %sx%s @ %.2ffps via %s/%s",
            camera_index,
            self._capture_handle.width,
            self._capture_handle.height,
            self.fps,
            self._capture_handle.backend_name,
            self._capture_handle.codec_name,
        )

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError

        if self._start is None:
            self._start = time.time()
        else:
            self._timestamp += int((1 / self.fps) * VIDEO_CLOCK_RATE)
            target_time = self._start + (self._timestamp / VIDEO_CLOCK_RATE)
            wait_time = target_time - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        success, frame = await asyncio.to_thread(self._capture.read)
        if not success:
            self.stop()
            raise MediaStreamError

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(rgb_frame, format="rgb24")
        video_frame.pts = self._timestamp
        video_frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        return video_frame

    def stop(self) -> None:
        if self._capture.isOpened():
            self._capture.release()
        super().stop()


async def generate_mjpeg_stream(settings: ServerSettings, camera_index: int):
    capture_handle = open_capture(camera_index, settings.width, settings.height, settings.fps)
    capture = capture_handle.capture
    frame_delay = 1 / max(1.0, min(capture_handle.fps or float(settings.fps), 15.0))

    try:
        while True:
            success, frame = await asyncio.to_thread(capture.read)
            if not success:
                break

            encoded, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
            if not encoded:
                await asyncio.sleep(frame_delay)
                continue

            jpeg_bytes = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii")
                + jpeg_bytes
                + b"\r\n"
            )
            await asyncio.sleep(frame_delay)
    finally:
        capture.release()


async def generate_body_skeleton_stream(settings: ServerSettings, camera_index: int):
    if mp is None:
        raise RuntimeError("mediapipe is not installed on the backend.")

    stream_width = min(settings.width, BODY_STREAM_MAX_WIDTH)
    stream_height = min(settings.height, BODY_STREAM_MAX_HEIGHT)
    stream_fps = min(settings.fps, BODY_STREAM_MAX_FPS)
    capture_handle = open_capture(camera_index, stream_width, stream_height, stream_fps)
    capture = capture_handle.capture
    renderer = BodySkeletonRenderer()
    frame_delay = 1 / max(1.0, min(capture_handle.fps or float(stream_fps), float(BODY_STREAM_MAX_FPS)))

    try:
        while True:
            success, frame = await asyncio.to_thread(capture.read)
            if not success:
                break

            processed_frame = frame
            largest_dimension = max(frame.shape[0], frame.shape[1])
            if largest_dimension > BODY_DETECTION_MAX_DIMENSION:
                scale = BODY_DETECTION_MAX_DIMENSION / float(largest_dimension)
                processed_frame = cv2.resize(
                    frame,
                    (
                        max(1, int(frame.shape[1] * scale)),
                        max(1, int(frame.shape[0] * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            skeleton_frame, pose_found = renderer.render(frame, processed_frame)
            if not pose_found:
                cv2.putText(
                    skeleton_frame,
                    "Stand inside the frame for skeleton tracking",
                    (28, max(48, skeleton_frame.shape[0] - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (156, 168, 182),
                    2,
                    cv2.LINE_AA,
                )

            encoded, jpeg = cv2.imencode(".jpg", skeleton_frame, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
            if not encoded:
                await asyncio.sleep(frame_delay)
                continue

            jpeg_bytes = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii")
                + jpeg_bytes
                + b"\r\n"
            )
            await asyncio.sleep(frame_delay)
    finally:
        renderer.close()
        capture.release()


async def wait_for_ice_gathering(pc: RTCPeerConnection, timeout: float = 5.0) -> None:
    if pc.iceGatheringState == "complete":
        return

    loop = asyncio.get_running_loop()
    done = loop.create_future()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering_state_change() -> None:
        if pc.iceGatheringState == "complete" and not done.done():
            done.set_result(True)

    try:
        await asyncio.wait_for(done, timeout=timeout)
    except asyncio.TimeoutError:
        LOGGER.warning("ICE gathering timed out; returning the best answer gathered so far.")


async def cleanup_peer_connection(app: FastAPI, pc: RTCPeerConnection) -> None:
    tracks = app.state.pc_tracks.pop(pc, [])
    for track in tracks:
        track.stop()
    if pc in app.state.pcs:
        app.state.pcs.remove(pc)
    await pc.close()


def build_status_page(settings: ServerSettings) -> str:
    camera_text = ", ".join(str(index) for index in settings.camera_indices)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI Healthcare FastAPI</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: #f4f8fc;
      color: #122136;
    }}
    main {{
      max-width: 840px;
      margin: 40px auto;
      padding: 0 20px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #d7e4f4;
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 18px 40px rgba(18, 33, 54, 0.08);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
    }}
    p {{
      margin: 0 0 18px;
      color: #5f738d;
      line-height: 1.7;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.8;
    }}
    code {{
      font-family: Consolas, monospace;
      background: #eff5fc;
      padding: 2px 6px;
      border-radius: 8px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>AI Healthcare FastAPI Backend</h1>
      <p>This backend serves the Android viewer over FastAPI and answers WebRTC offers directly.</p>
      <ul>
        <li>Status route: <code>GET /health</code></li>
        <li>Offer route: <code>POST /offer</code></li>
        <li>Face fallback stream: <code>GET /mjpeg</code></li>
        <li>Body skeleton stream: <code>GET /body-skeleton.mjpeg</code></li>
        <li>Listening address: <code>http://{settings.host}:{settings.port}</code></li>
        <li>Configured cameras: <code>{camera_text}</code></li>
        <li>Capture target: <code>{settings.width}x{settings.height} @ {settings.fps}fps</code></li>
        <li>Capture mode: <code>best available resolution with MJPG priority</code></li>
      </ul>
    </section>
  </main>
</body>
</html>"""


def load_viewer_html() -> str:
    viewer_path = Path(__file__).resolve().parents[2] / "Android" / "app" / "src" / "main" / "assets" / "viewer.html"
    return viewer_path.read_text(encoding="utf-8")


def create_app(settings: ServerSettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.pcs = set()
        app.state.pc_tracks = {}
        yield
        await asyncio.gather(
            *(cleanup_peer_connection(app, pc) for pc in list(app.state.pcs)),
            return_exceptions=True,
        )

    app = FastAPI(
        title="AI Healthcare FastAPI Backend",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        return HTMLResponse(build_status_page(settings))

    @app.get("/viewer", response_class=HTMLResponse)
    async def viewer() -> HTMLResponse:
        return HTMLResponse(load_viewer_html())

    @app.get("/health")
    @app.get("/internal/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "configuredCameras": settings.camera_indices,
            "host": settings.host,
            "port": settings.port,
            "server": "fastapi",
            "bodySkeletonAvailable": mp is not None,
            "captureRequest": {
                "width": settings.width,
                "height": settings.height,
                "fps": settings.fps,
                "mode": "best_available",
            },
        }

    @app.get("/mjpeg")
    async def mjpeg(camera: int | None = None) -> StreamingResponse:
        camera_index = settings.camera_indices[0] if camera is None else camera
        if camera_index not in settings.camera_indices:
            raise HTTPException(status_code=404, detail="Camera index is not configured on the server.")

        return StreamingResponse(
            generate_mjpeg_stream(settings, camera_index),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.get("/body-skeleton.mjpeg")
    async def body_skeleton_mjpeg(camera: int | None = None) -> StreamingResponse:
        if mp is None:
            raise HTTPException(status_code=503, detail="mediapipe is not installed on the backend.")

        camera_index = settings.camera_indices[0] if camera is None else camera
        if camera_index not in settings.camera_indices:
            raise HTTPException(status_code=404, detail="Camera index is not configured on the server.")

        return StreamingResponse(
            generate_body_skeleton_stream(settings, camera_index),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.post("/offer")
    @app.post("/internal/offer")
    async def offer(payload: OfferPayload, request: Request) -> dict[str, str]:
        requested_tracks = payload.requestedTracks or len(settings.camera_indices)
        requested_tracks = max(1, requested_tracks)
        offer_description = RTCSessionDescription(sdp=payload.sdp, type=payload.type)

        pc = RTCPeerConnection()
        request.app.state.pcs.add(pc)
        request.app.state.pc_tracks[pc] = []

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            LOGGER.info("Peer connection state changed to %s", pc.connectionState)
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await cleanup_peer_connection(request.app, pc)

        LOGGER.info("Incoming offer. Preparing up to %s video tracks.", requested_tracks)
        await pc.setRemoteDescription(offer_description)

        for camera_index in settings.camera_indices[:requested_tracks]:
            track = OpenCVCameraTrack(
                camera_index=camera_index,
                width=settings.width,
                height=settings.height,
                fps=settings.fps,
            )
            request.app.state.pc_tracks[pc].append(track)
            pc.addTrack(track)
            LOGGER.info("Attached camera index %s to the peer connection.", camera_index)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await wait_for_ice_gathering(pc)

        local_description = pc.localDescription
        if local_description is None:
            await cleanup_peer_connection(request.app, pc)
            raise RuntimeError("Local description was not created.")

        return {
            "sdp": local_description.sdp,
            "type": local_description.type,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastAPI backend for the AI healthcare Android viewer."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address.")
    parser.add_argument("--port", default=8080, type=int, help="Server port.")
    parser.add_argument(
        "--camera",
        action="append",
        type=int,
        default=None,
        help="Camera index to stream. Repeat this flag for multiple cameras.",
    )
    parser.add_argument("--width", default=DEFAULT_CAPTURE_WIDTH, type=int, help="Capture width.")
    parser.add_argument("--height", default=DEFAULT_CAPTURE_HEIGHT, type=int, help="Capture height.")
    parser.add_argument("--fps", default=DEFAULT_CAPTURE_FPS, type=int, help="Capture frame rate.")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe local camera indices and exit.",
    )
    parser.add_argument(
        "--probe-limit",
        default=10,
        type=int,
        help="Highest camera index to probe when --list-cameras is used.",
    )
    args = parser.parse_args()
    if not args.camera:
        args.camera = [0]
    return args


def main() -> None:
    args = parse_args()
    args.camera = list(dict.fromkeys(args.camera))

    if args.list_cameras:
        available = probe_cameras(args.probe_limit)
        print(json.dumps({"availableCameraIndices": available}, ensure_ascii=False, indent=2))
        return

    settings = ServerSettings(
        host=args.host,
        port=args.port,
        camera_indices=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    LOGGER.info("Configured camera indices: %s", settings.camera_indices)
    LOGGER.info("Starting FastAPI backend on http://%s:%s", settings.host, settings.port)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
