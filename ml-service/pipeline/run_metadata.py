import json
import time
from pathlib import Path
from typing import Any

from pipeline.config import RUN_METADATA_FILENAME


def write_run_metadata(
    run_dir: str | Path,
    *,
    run_type: str,
    model_path: str | Path | None = None,
    metrics: dict[str, Any] | None = None,
    prepare_summary: dict[str, Any] | None = None,
    best_summary: dict[str, Any] | None = None,
    embed_summary: dict[str, Any] | None = None,
    index_summary: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    promoted: bool = False,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)

    metadata = {
        "run_type": run_type,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "model_path": str(model_path) if model_path is not None else None,
        "promoted": promoted,
        "warnings": warnings or [],
        "payload": payload or {},
        "prepare_summary": prepare_summary,
        "best_summary": best_summary,
        "metrics": metrics,
        "embed_summary": embed_summary,
        "index_summary": index_summary,
    }

    path = run_dir / RUN_METADATA_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata