import json
import logging
import shutil
from pathlib import Path

import numpy as np

from db.track_repo import load_tracks
from pipeline.build_index import append_to_faiss_index
from pipeline.config import (
    MODEL_PATH,
    EMBEDDINGS_PATH,
    SONG_IDS_PATH,
    FAISS_INDEX_PATH,
    SONG_MANIFEST_PATH,
    METRICS_PATH,
)
from pipeline.prepare_data import prepare_data
from pipeline.build_embeddings_from_prepared import build_embeddings_from_prepared
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
        raise ValueError("Embedding dimensions differ")

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
) -> None:
    with open(existing_manifest_path, "r", encoding="utf-8") as f:
        old_rows = json.load(f)

    with open(new_manifest_path, "r", encoding="utf-8") as f:
        new_rows = json.load(f)

    merged = list(old_rows) + list(new_rows)

    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def build_incremental_run(
    bucket: str,
    prefix: str,
    run_dir: str | Path,
) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    new_track_ids = detect_new_track_ids(bucket=bucket, prefix=prefix)
    if not new_track_ids:
        logger.info("Incremental sync found no new tracks")
        return {
            "new_track_ids": [],
            "new_tracks": 0,
            "new_vectors": 0,
            "no_changes": True,
        }

    prepare_summary = prepare_data(
        bucket=bucket,
        prefix=prefix,
        only_track_ids=set(new_track_ids),
        append=True,
    )

    # Use currently active model, do not retrain
    shutil.copy2(MODEL_PATH, run_dir / "model.pt")

    if METRICS_PATH.exists():
        shutil.copy2(METRICS_PATH, run_dir / "metrics.json")
    else:
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"mode": "incremental-copy"}, f, ensure_ascii=False, indent=2)

    new_embeddings_path = run_dir / "new_embeddings.npy"
    new_song_ids_path = run_dir / "new_song_ids.npy"
    new_manifest_path = run_dir / "new_song_manifest.json"

    embed_summary = build_embeddings_from_prepared(
        model_path=MODEL_PATH,
        embeddings_out=new_embeddings_path,
        song_ids_out=new_song_ids_path,
        manifest_out=new_manifest_path,
        only_track_ids=set(new_track_ids),
    )

    _append_float32_embeddings(
        existing_path=EMBEDDINGS_PATH,
        new_path=new_embeddings_path,
        output_path=run_dir / "embeddings.npy",
    )

    _append_int64_song_ids(
        existing_path=SONG_IDS_PATH,
        new_path=new_song_ids_path,
        output_path=run_dir / "song_ids.npy",
    )

    _append_song_manifests(
        existing_manifest_path=SONG_MANIFEST_PATH,
        new_manifest_path=new_manifest_path,
        output_manifest_path=run_dir / "song_manifest.json",
    )

    append_to_faiss_index(
        base_index_path=FAISS_INDEX_PATH,
        new_embeddings_path=new_embeddings_path,
        index_out_path=run_dir / "faiss.index",
    )

    return {
        "new_track_ids": new_track_ids,
        "new_tracks": len(new_track_ids),
        "new_vectors": int(embed_summary["vectors"]),
        "prepare_summary": prepare_summary,
        "embed_summary": embed_summary,
        "no_changes": False,
    }