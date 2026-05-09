import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUS_PATH = Path("/app/artifacts/fingerprint-experiments/job_status.json")

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
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with STATUS_PATH.open("w", encoding="utf-8") as f:
        json.dump(_status, f, ensure_ascii=False, indent=2)


def update_status(**fields) -> dict:
    with _lock:
        _status.update(fields)
        _persist()
        return dict(_status)


def get_status() -> dict:
    with _lock:
        return dict(_status)