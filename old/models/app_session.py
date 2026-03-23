from dataclasses import dataclass
from typing import Optional


@dataclass
class AppSession:
    gpt_api_key: Optional[str] = None
    captured_face_path: Optional[str] = None
    captured_tongue_path: Optional[str] = None
    camera_finished: bool = False
    analysis_complete: bool = False

    def set_api_key(self, api_key: str) -> None:
        self.gpt_api_key = api_key

    def set_captured_images(self, face_path: str, tongue_path: str) -> None:
        self.captured_face_path = face_path
        self.captured_tongue_path = tongue_path
        self.camera_finished = True
        self.analysis_complete = False

    def reset_capture(self) -> None:
        self.captured_face_path = None
        self.captured_tongue_path = None
        self.camera_finished = False
        self.analysis_complete = False

    def mark_analysis_complete(self) -> None:
        self.analysis_complete = True

