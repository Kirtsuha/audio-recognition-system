import os
import shutil
import time
from pathlib import Path
from typing import Any

from pipeline.config import ACTIVE_ARTIFACTS_DIR, RUNS_DIR


def _get_active_run_dir() -> Path | None:
    pointer = ACTIVE_ARTIFACTS_DIR / "active_run.json"
    if not pointer.exists():
        return None

    import json

    with pointer.open("r", encoding="utf-8") as f:
        data = json.load(f)

    run_dir = data.get("run_dir")
    return Path(run_dir) if run_dir else None


def _run_sort_key(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _is_failed_run(run_dir: Path) -> bool:

    required_any = ["model.pt", "metrics.json", "faiss.index"]
    return not any((run_dir / name).exists() for name in required_any)


def _delete_path(path: Path, dry_run: bool) -> int:
    if not path.exists():
        return 0

    if path.is_file():
        size = path.stat().st_size
        if not dry_run:
            path.unlink()
        return size

    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass

    if not dry_run:
        shutil.rmtree(path)

    return total


def cleanup_runs(
    *,
    dry_run: bool = True,
    keep_last_full_runs: int = 2,
    keep_last_experiment_runs: int = 5,
    keep_last_incremental_runs: int = 2,
    delete_failed_runs: bool = True,
    delete_epoch_checkpoints: bool = True,
    delete_eval_results: bool = False,
    min_age_hours: int = 24,
) -> dict[str, Any]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    active_run_dir = _get_active_run_dir()
    now = time.time()
    min_age_sec = min_age_hours * 3600

    runs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]

    full_runs = sorted([p for p in runs if p.name.startswith("full_")], key=_run_sort_key, reverse=True)
    exp_runs = sorted([p for p in runs if p.name.startswith("exp_")], key=_run_sort_key, reverse=True)
    inc_runs = sorted([p for p in runs if p.name.startswith("incremental_")], key=_run_sort_key, reverse=True)

    keep = set(full_runs[:keep_last_full_runs])
    keep.update(exp_runs[:keep_last_experiment_runs])
    keep.update(inc_runs[:keep_last_incremental_runs])

    if active_run_dir is not None:
        keep.add(active_run_dir)

    deleted = []
    skipped = []
    freed_bytes = 0

    for run_dir in runs:
        try:
            age_sec = now - run_dir.stat().st_mtime
        except OSError:
            continue

        if age_sec < min_age_sec:
            skipped.append({"path": str(run_dir), "reason": "too_young"})
            continue

        if run_dir in keep:
            skipped.append({"path": str(run_dir), "reason": "kept"})
            continue

        failed = _is_failed_run(run_dir)

        if failed and not delete_failed_runs:
            skipped.append({"path": str(run_dir), "reason": "failed_run_delete_disabled"})
            continue

        size = _delete_path(run_dir, dry_run=dry_run)
        freed_bytes += size
        deleted.append(
            {
                "path": str(run_dir),
                "bytes": size,
                "failed": failed,
                "type": "run_dir",
            }
        )

    checkpoint_deleted = []

    if delete_epoch_checkpoints:
        for run_dir in keep:
            if not run_dir.exists():
                continue

            has_final_model = (run_dir / "model.pt").exists() or (run_dir / "model_best.pt").exists()

            if not has_final_model:
                continue

            for path in run_dir.glob("model_epoch_*.pt"):
                size = _delete_path(path, dry_run=dry_run)
                freed_bytes += size
                checkpoint_deleted.append({"path": str(path), "bytes": size})

    eval_deleted = []

    if delete_eval_results:
        for run_dir in keep:
            if not run_dir.exists():
                continue

            for path in run_dir.glob("eval_results*.jsonl"):
                size = _delete_path(path, dry_run=dry_run)
                freed_bytes += size
                eval_deleted.append({"path": str(path), "bytes": size})

    return {
        "dry_run": dry_run,
        "freed_bytes": freed_bytes,
        "freed_gb": freed_bytes / (1024 ** 3),
        "deleted": deleted,
        "checkpoint_deleted": checkpoint_deleted,
        "eval_deleted": eval_deleted,
        "skipped": skipped,
        "active_run_dir": str(active_run_dir) if active_run_dir else None,
    }