import json
from pathlib import Path
from typing import Any

from pipeline.config import (
    ACTIVE_ARTIFACTS_DIR,
    FULL_RETRAIN_NEW_TRACK_MIN,
    FULL_RETRAIN_NEW_TRACK_RATIO,
    RUN_METADATA_FILENAME,
    SONG_MANIFEST_PATH,
)
from pipeline.manifest import load_all_prepared_manifest


def _load_active_song_manifest_count() -> int:
    if not SONG_MANIFEST_PATH.exists():
        return 0

    with SONG_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    track_ids = {int(row["track_id"]) for row in rows}
    return len(track_ids)


def _load_active_metadata() -> dict[str, Any]:
    active_pointer = ACTIVE_ARTIFACTS_DIR / "active_run.json"

    if not active_pointer.exists():
        return {}

    with active_pointer.open("r", encoding="utf-8") as f:
        pointer = json.load(f)

    run_dir = Path(pointer["run_dir"])
    metadata_path = run_dir / RUN_METADATA_FILENAME

    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_retrain_status() -> dict[str, Any]:
    prepared_tracks = len({int(row["track_id"]) for row in load_all_prepared_manifest()})
    active_index_tracks = _load_active_song_manifest_count()
    active_meta = _load_active_metadata()

    train_limit = 0
    payload = active_meta.get("payload") or {}

    if isinstance(payload, dict):
        train_limit = int(payload.get("train_limit") or 0)

    last_full_index_tracks = active_index_tracks
    index_summary = active_meta.get("index_summary") or {}
    embed_summary = active_meta.get("embed_summary") or {}

    if isinstance(embed_summary, dict):
        last_full_index_tracks = int(embed_summary.get("tracks") or active_index_tracks)
    elif isinstance(index_summary, dict):
        last_full_index_tracks = active_index_tracks

    new_tracks_since_active = max(prepared_tracks - active_index_tracks, 0)
    new_track_ratio = (
        new_tracks_since_active / max(active_index_tracks, 1)
        if active_index_tracks > 0
        else 1.0 if prepared_tracks > 0 else 0.0
    )

    reasons = []

    if new_tracks_since_active >= FULL_RETRAIN_NEW_TRACK_MIN:
        reasons.append(
            f"new_tracks_since_active >= FULL_RETRAIN_NEW_TRACK_MIN "
            f"({new_tracks_since_active} >= {FULL_RETRAIN_NEW_TRACK_MIN})"
        )

    if new_track_ratio >= FULL_RETRAIN_NEW_TRACK_RATIO:
        reasons.append(
            f"new_track_ratio >= FULL_RETRAIN_NEW_TRACK_RATIO "
            f"({new_track_ratio:.4f} >= {FULL_RETRAIN_NEW_TRACK_RATIO})"
        )

    return {
        "prepared_tracks": prepared_tracks,
        "active_index_tracks": active_index_tracks,
        "last_full_index_tracks": last_full_index_tracks,
        "last_train_limit": train_limit,
        "new_tracks_since_active": new_tracks_since_active,
        "new_track_ratio": new_track_ratio,
        "retrain_recommended": bool(reasons),
        "reasons": reasons,
        "thresholds": {
            "full_retrain_new_track_min": FULL_RETRAIN_NEW_TRACK_MIN,
            "full_retrain_new_track_ratio": FULL_RETRAIN_NEW_TRACK_RATIO,
        },
    }