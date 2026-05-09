import json
import logging
import shutil
from pathlib import Path

import numpy as np

from db.track_repo import load_tracks
from pipeline.build_embeddings_from_prepared import build_embeddings_from_prepared
from pipeline.build_index import build_faiss_index
from pipeline.config import (
    MODEL_PATH,
    EMBEDDINGS_PATH,
    SONG_IDS_PATH,
    SONG_MANIFEST_PATH,
    METRICS_PATH,
    FAISS_INDEX_META_PATH,
)
from pipeline.prepare_data import prepare_data
from pipeline.retrain_status import get_retrain_status
from pipeline.run_metadata import write_run_metadata
from s3.s3_list import list_all_songs

logger = logging.getLogger("ml-pipeline.incremental")


def load_active_track_ids() -> set[int]:
    if not SONG_MANIFEST_PATH.exists():
        return set()

    with SONG_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    return {int(row["track_id"]) for row in rows}


def detect_new_track_ids(bucket: str, prefix: str) -> list[int]:
    active_track_ids = load_active_track_ids()
    s3_keys = set(list_all_songs(bucket=bucket, prefix=prefix))
    db_tracks = load_tracks()

    new_ids = []

    for row in db_tracks:
        track_id = int(row["track_id"])
        s3_key = row["s3_key"]

        if s3_key not in s3_keys:
            continue

        if track_id in active_track_ids:
            continue

        new_ids.append(track_id)

    new_ids.sort()

    logger.info(
        "Incremental detect-new-tracks active=%s db=%s new=%s",
        len(active_track_ids),
        len(db_tracks),
        len(new_ids),
    )

    return new_ids


def _append_float32_embeddings(
    existing_path: str | Path,
    new_path: str | Path,
    output_path: str | Path,
) -> None:
    old = np.load(existing_path, mmap_mode="r")
    new = np.load(new_path, mmap_mode="r")

    if old.ndim != 2 or new.ndim != 2:
        raise ValueError("Expected both embeddings arrays to be 2D")

    if old.shape[1] != new.shape[1]:
        raise ValueError(f"Embedding dimensions differ: old={old.shape}, new={new.shape}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(old.shape[0] + new.shape[0], old.shape[1]),
    )
    out[: old.shape[0]] = old[:]
    out[old.shape[0]:] = new[:]
    del out


def _append_int64_song_ids(
    existing_path: str | Path,
    new_path: str | Path,
    output_path: str | Path,
) -> None:
    old = np.load(existing_path, mmap_mode="r")
    new = np.load(new_path, mmap_mode="r")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.int64,
        shape=(old.shape[0] + new.shape[0],),
    )
    out[: old.shape[0]] = old[:]
    out[old.shape[0]:] = new[:]
    del out


def _append_song_manifests(
    existing_manifest_path: str | Path,
    new_manifest_path: str | Path,
    output_manifest_path: str | Path,
) -> int:
    with open(existing_manifest_path, "r", encoding="utf-8") as f:
        old_rows = json.load(f)

    with open(new_manifest_path, "r", encoding="utf-8") as f:
        new_rows = json.load(f)

    merged = list(old_rows) + list(new_rows)

    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    return len({int(row["track_id"]) for row in merged})


def build_incremental_run(
    bucket: str,
    prefix: str,
    run_dir: str | Path,
    *,
    max_new_tracks: int = 0,
    allow_incremental_when_retrain_recommended: bool = True,
    index_windows_override: int | None = 96,
) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Active model is missing: {MODEL_PATH}")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Active embeddings are missing: {EMBEDDINGS_PATH}")
    if not SONG_IDS_PATH.exists():
        raise FileNotFoundError(f"Active song_ids are missing: {SONG_IDS_PATH}")
    if not SONG_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Active song_manifest is missing: {SONG_MANIFEST_PATH}")

    new_track_ids = detect_new_track_ids(bucket=bucket, prefix=prefix)

    if max_new_tracks > 0:
        new_track_ids = new_track_ids[:max_new_tracks]

    retrain_status = get_retrain_status()

    if retrain_status["retrain_recommended"] and not allow_incremental_when_retrain_recommended:
        return {
            "no_changes": True,
            "blocked": True,
            "reason": "retrain_recommended",
            "retrain_status": retrain_status,
            "new_tracks": len(new_track_ids),
        }

    if not new_track_ids:
        logger.info("Incremental sync found no new tracks")
        return {
            "new_track_ids": [],
            "new_tracks": 0,
            "new_vectors": 0,
            "no_changes": True,
            "retrain_status": retrain_status,
        }

    prepare_summary = prepare_data(
        bucket=bucket,
        prefix=prefix,
        only_track_ids=set(new_track_ids),
        append=True,
        skip_existing=True,
    )

    shutil.copy2(MODEL_PATH, run_dir / "model.pt")

    if METRICS_PATH.exists():
        shutil.copy2(METRICS_PATH, run_dir / "metrics.json")
    else:
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"mode": "incremental-copy"}, f, ensure_ascii=False, indent=2)

    if FAISS_INDEX_META_PATH.exists():
        shutil.copy2(FAISS_INDEX_META_PATH, run_dir / "faiss_meta.source.json")

    new_embeddings_path = run_dir / "new_embeddings.npy"
    new_song_ids_path = run_dir / "new_song_ids.npy"
    new_manifest_path = run_dir / "new_song_manifest.json"

    embed_summary = build_embeddings_from_prepared(
        model_path=MODEL_PATH,
        embeddings_out=new_embeddings_path,
        song_ids_out=new_song_ids_path,
        manifest_out=new_manifest_path,
        only_track_ids=set(new_track_ids),
        index_windows_override=index_windows_override,
        index_all_prepared=True,
    )

    merged_embeddings_path = run_dir / "embeddings.npy"
    merged_song_ids_path = run_dir / "song_ids.npy"
    merged_manifest_path = run_dir / "song_manifest.json"

    _append_float32_embeddings(
        existing_path=EMBEDDINGS_PATH,
        new_path=new_embeddings_path,
        output_path=merged_embeddings_path,
    )

    _append_int64_song_ids(
        existing_path=SONG_IDS_PATH,
        new_path=new_song_ids_path,
        output_path=merged_song_ids_path,
    )

    total_tracks = _append_song_manifests(
        existing_manifest_path=SONG_MANIFEST_PATH,
        new_manifest_path=new_manifest_path,
        output_manifest_path=merged_manifest_path,
    )

    index_path = run_dir / "faiss.index"
    index_meta_path = run_dir / "faiss_meta.json"

    index_summary = build_faiss_index(
        embeddings_path=merged_embeddings_path,
        song_ids_path=merged_song_ids_path,
        index_out_path=index_path,
        meta_out_path=index_meta_path,
    )

    summary = {
        "new_track_ids": new_track_ids,
        "new_tracks": len(new_track_ids),
        "new_vectors": int(embed_summary["vectors"]),
        "total_tracks": total_tracks,
        "prepare_summary": prepare_summary,
        "embed_summary": embed_summary,
        "index_summary": index_summary,
        "retrain_status": retrain_status,
        "no_changes": False,
        "blocked": False,
    }

    write_run_metadata(
        run_dir=run_dir,
        run_type="incremental-sync",
        model_path=run_dir / "model.pt",
        prepare_summary=prepare_summary,
        embed_summary=embed_summary,
        index_summary=index_summary,
        payload={
            "bucket": bucket,
            "prefix": prefix,
            "max_new_tracks": max_new_tracks,
            "index_windows_override": index_windows_override,
        },
        warnings=retrain_status.get("reasons", []),
    )

    return summary