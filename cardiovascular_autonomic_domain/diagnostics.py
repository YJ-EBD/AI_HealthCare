from __future__ import annotations

import json
import logging
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


DOMAIN_DIR = Path(__file__).resolve().parent
APP_DATA_DIR = DOMAIN_DIR / "app_data"
LOGS_DIR = APP_DATA_DIR / "logs"
LOG_PATH = LOGS_DIR / "healthcare.log"

LOGGER_NAME = "healthcare_app"
_configured = False


def ensure_runtime_dirs() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging() -> Path:
    global _configured
    ensure_runtime_dirs()
    if _configured:
        return LOG_PATH

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(LOG_PATH, maxBytes=1_500_000, backupCount=5, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _configured = True
    return LOG_PATH


def get_logger() -> logging.Logger:
    configure_logging()
    return logging.getLogger(LOGGER_NAME)


def _format_details(details: dict[str, Any] | None) -> str:
    if not details:
        return ""
    try:
        return " | " + json.dumps(details, ensure_ascii=False, sort_keys=True)
    except TypeError:
        safe_details = {key: str(value) for key, value in details.items()}
        return " | " + json.dumps(safe_details, ensure_ascii=False, sort_keys=True)


def log_event(source: str, message: str, *, level: str = "info", details: dict[str, Any] | None = None) -> Path:
    logger = get_logger()
    payload = f"{source} | {message}{_format_details(details)}"
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(payload)
    return LOG_PATH


def log_exception(source: str, exc: BaseException, *, details: dict[str, Any] | None = None) -> Path:
    merged_details = dict(details or {})
    merged_details["exception_type"] = type(exc).__name__
    merged_details["traceback"] = traceback.format_exc()
    return log_event(source, str(exc), level="error", details=merged_details)
