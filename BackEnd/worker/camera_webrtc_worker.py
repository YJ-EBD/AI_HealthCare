import argparse
import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from fractions import Fraction

import cv2
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("fastapi_camera_backend")

VIDEO_CLOCK_RATE = 90000


@dataclass(slots=True)
class ServerSettings:
    host: str
    port: int
    camera_indices: list[int]
    width: int
    height: int
    fps: int


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


def open_capture(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    for backend_name, backend in candidate_backends():
        capture = cv2.VideoCapture(index, backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        if not capture.isOpened():
            capture.release()
            continue

        success, _ = capture.read()
        if success:
            LOGGER.info("Camera %s opened with backend %s", index, backend_name)
            return capture

        capture.release()

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
        self.fps = fps
        self._capture = open_capture(camera_index, width, height, fps)
        self._timestamp = 0
        self._start: float | None = None

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
        <li>Listening address: <code>http://{settings.host}:{settings.port}</code></li>
        <li>Configured cameras: <code>{camera_text}</code></li>
        <li>Capture: <code>{settings.width}x{settings.height} @ {settings.fps}fps</code></li>
      </ul>
    </section>
  </main>
</body>
</html>"""


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

    @app.get("/health")
    @app.get("/internal/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "configuredCameras": settings.camera_indices,
            "host": settings.host,
            "port": settings.port,
            "server": "fastapi",
        }

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
    parser.add_argument("--width", default=640, type=int, help="Capture width.")
    parser.add_argument("--height", default=480, type=int, help="Capture height.")
    parser.add_argument("--fps", default=15, type=int, help="Capture frame rate.")
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
