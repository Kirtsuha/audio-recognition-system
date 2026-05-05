import json
import shutil
from pathlib import Path

from pipeline.config import ACTIVE_ARTIFACTS_DIR


def validate_run_artifacts(run_dir: str | Path, require_metrics: bool = True) -> None:
    run_dir = Path(run_dir)

    required = [
        "model.pt",
        "embeddings.npy",
        "song_ids.npy",
        "faiss.index",
        "song_manifest.json",
    ]

    if require_metrics:
        required.append("metrics.json")

    for name in required:
        src = run_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing artifact: {src}")


def promote_run_to_active(run_dir: str | Path, require_metrics: bool = True) -> None:
    run_dir = Path(run_dir)
    validate_run_artifacts(run_dir, require_metrics=require_metrics)

    ACTIVE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        "model.pt",
        "embeddings.npy",
        "song_ids.npy",
        "faiss.index",
        "song_manifest.json",
    ]

    if require_metrics:
        required.append("metrics.json")

    optional = [
        "faiss_meta.json",
        "run_metadata.json",
        "threshold_tuning.json",
        "reranker_ref_embeddings.npy",
        "reranker_ref_meta.jsonl",
        "reranker_ref_config.json",
    ]

    for name in required:
        shutil.copy2(run_dir / name, ACTIVE_ARTIFACTS_DIR / name)

    for name in optional:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, ACTIVE_ARTIFACTS_DIR / name)

    pointer = ACTIVE_ARTIFACTS_DIR / "active_run.json"
    with pointer.open("w", encoding="utf-8") as f:
        json.dump({"run_dir": str(run_dir)}, f, ensure_ascii=False, indent=2)