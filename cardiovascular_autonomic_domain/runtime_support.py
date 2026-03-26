from __future__ import annotations

import json
import logging
import sqlite3
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


DOMAIN_DIR = Path(__file__).resolve().parent
APP_DATA_DIR = DOMAIN_DIR / "app_data"
LOGS_DIR = APP_DATA_DIR / "logs"
LOG_PATH = LOGS_DIR / "healthcare.log"
DB_PATH = APP_DATA_DIR / "healthcare.db"

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


def _utc_now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ProductStore:
    def __init__(self, db_path: Path | None = None) -> None:
        ensure_runtime_dirs()
        self.db_path = Path(db_path or DB_PATH)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER,
                    sex TEXT NOT NULL DEFAULT 'unknown',
                    calibration_sbp REAL,
                    calibration_dbp REAL,
                    notes TEXT NOT NULL DEFAULT '',
                    is_default INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    profile_id INTEGER,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    port TEXT,
                    duration_s REAL,
                    output_dir TEXT,
                    report_path TEXT,
                    summary_path TEXT,
                    signal_quality_score REAL,
                    overall_confidence_score REAL,
                    heart_rate_bpm REAL,
                    analysis_mode_label TEXT,
                    no_read_outputs TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(profile_id) REFERENCES profiles(id)
                );

                CREATE TABLE IF NOT EXISTS diagnostics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '',
                    session_id INTEGER,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                """
            )

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM profiles
                ORDER BY is_default DESC, last_used_at DESC NULLS LAST, updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_profile(self, profile_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        return dict(row) if row else None

    def get_default_profile(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM profiles WHERE is_default = 1 ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def upsert_profile(
        self,
        *,
        name: str,
        age: int | None,
        sex: str,
        calibration_sbp: float | None,
        calibration_dbp: float | None,
        notes: str = "",
        profile_id: int | None = None,
        make_default: bool = False,
    ) -> dict[str, Any]:
        now_text = _utc_now_text()
        with self._connect() as connection:
            if make_default:
                connection.execute("UPDATE profiles SET is_default = 0")

            if profile_id is None:
                cursor = connection.execute(
                    """
                    INSERT INTO profiles (
                        name, age, sex, calibration_sbp, calibration_dbp, notes,
                        is_default, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        age,
                        sex,
                        calibration_sbp,
                        calibration_dbp,
                        notes,
                        1 if make_default else 0,
                        now_text,
                        now_text,
                    ),
                )
                profile_id = int(cursor.lastrowid)
            else:
                connection.execute(
                    """
                    UPDATE profiles
                    SET
                        name = ?,
                        age = ?,
                        sex = ?,
                        calibration_sbp = ?,
                        calibration_dbp = ?,
                        notes = ?,
                        is_default = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        name,
                        age,
                        sex,
                        calibration_sbp,
                        calibration_dbp,
                        notes,
                        1 if make_default else 0,
                        now_text,
                        profile_id,
                    ),
                )

            row = connection.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        return dict(row) if row else {}

    def mark_profile_used(self, profile_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE profiles SET last_used_at = ?, updated_at = ? WHERE id = ?",
                (_utc_now_text(), _utc_now_text(), profile_id),
            )

    def record_session(
        self,
        *,
        profile_id: int | None,
        status: str,
        mode: str,
        port: str | None,
        duration_s: float | None,
        output_dir: str | None,
        report_path: str | None,
        summary_path: str | None,
        signal_quality_score: float | None,
        overall_confidence_score: float | None,
        heart_rate_bpm: float | None,
        analysis_mode_label: str | None,
        no_read_outputs: list[str] | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sessions (
                    created_at, profile_id, status, mode, port, duration_s, output_dir,
                    report_path, summary_path, signal_quality_score, overall_confidence_score,
                    heart_rate_bpm, analysis_mode_label, no_read_outputs
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now_text(),
                    profile_id,
                    status,
                    mode,
                    port,
                    duration_s,
                    output_dir,
                    report_path,
                    summary_path,
                    signal_quality_score,
                    overall_confidence_score,
                    heart_rate_bpm,
                    analysis_mode_label,
                    json.dumps(no_read_outputs or [], ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    sessions.*,
                    profiles.name AS profile_name
                FROM sessions
                LEFT JOIN profiles ON profiles.id = sessions.profile_id
                ORDER BY sessions.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["no_read_outputs"] = json.loads(item.get("no_read_outputs") or "[]")
            except json.JSONDecodeError:
                item["no_read_outputs"] = []
            items.append(item)
        return items

    def record_diagnostic(
        self,
        *,
        level: str,
        source: str,
        message: str,
        details: dict[str, Any] | None = None,
        session_id: int | None = None,
    ) -> int:
        details_json = json.dumps(details or {}, ensure_ascii=False)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO diagnostics (created_at, level, source, message, details_json, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (_utc_now_text(), level, source, message, details_json, session_id),
            )
            return int(cursor.lastrowid)

    def list_diagnostics(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM diagnostics
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_settings(self, key: str, value: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO app_settings (key, value_json)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                """,
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def load_settings(self, key: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT value_json FROM app_settings WHERE key = ?", (key,)).fetchone()
        if not row:
            return {}
        try:
            value = json.loads(str(row["value_json"]))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
