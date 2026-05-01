import json
import shutil
from pathlib import Path

from pipeline.config import ACTIVE_ARTIFACTS_DIR


def promote_run_to_active(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    ACTIVE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        "model.pt",
        "embeddings.npy",
        "song_ids.npy",
        "faiss.index",
        "song_manifest.json",
        "metrics.json",
    ]

    optional = [
        "faiss_meta.json",
    ]

    for name in required:
        src = run_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing artifact for promotion: {src}")

    for name in required:
        shutil.copy2(run_dir / name, ACTIVE_ARTIFACTS_DIR / name)

    for name in optional:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, ACTIVE_ARTIFACTS_DIR / name)

    pointer = ACTIVE_ARTIFACTS_DIR / "active_run.json"
    with pointer.open("w", encoding="utf-8") as f:
        json.dump({"run_dir": str(run_dir)}, f, ensure_ascii=False, indent=2)