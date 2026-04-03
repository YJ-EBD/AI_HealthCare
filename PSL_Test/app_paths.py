from __future__ import annotations

from datetime import datetime
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"
APP_DATA_DIR = APP_DIR / "app_data"
LOG_PATH = APP_DATA_DIR / "healthcare.log"


def ensure_runtime_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)


def new_session_dir(prefix: str = "session") -> Path:
    ensure_runtime_dirs()
    session_dir = OUTPUTS_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def append_runtime_log(message: str) -> None:
    ensure_runtime_dirs()
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
