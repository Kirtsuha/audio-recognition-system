import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import JOB_STATUS_PATH


_lock = threading.Lock()

_status: dict[str, Any] = {
    "current_job": None,
    "phase": None,
    "started_at": None,
    "finished_at": None,
    "last_error": None,
    "last_success": None,
    "progress": {},
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persist() -> None:
    JOB_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOB_STATUS_PATH.open("w", encoding="utf-8") as f:
        json.dump(_status, f, ensure_ascii=False, indent=2)


def load_status_from_disk() -> dict[str, Any]:
    with _lock:
        if JOB_STATUS_PATH.exists():
            with JOB_STATUS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                _status.update(data)
        return dict(_status)


def update_status(**fields) -> dict[str, Any]:
    with _lock:
        _status.update(fields)
        _persist()
        return dict(_status)


def update_progress(**fields) -> dict[str, Any]:
    with _lock:
        progress = dict(_status.get("progress", {}))
        progress.update(fields)
        _status["progress"] = progress
        _persist()
        return dict(_status)


def reset_progress() -> dict[str, Any]:
    with _lock:
        _status["progress"] = {}
        _persist()
        return dict(_status)


def get_status() -> dict[str, Any]:
    with _lock:
        return dict(_status)