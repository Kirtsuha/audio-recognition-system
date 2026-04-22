import json
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from db.track_repo import load_tracks
from pipeline.config import PREPARED_TRACKS_DIR, PREPARED_MANIFEST_PATH, INVALID_KEYS_PATH
from pipeline.dataset import assign_split, load_audio
from s3.s3_list import list_all_songs
from s3.s3_loader import download_song

logger = logging.getLogger("ml-pipeline.prepare-data")


def _load_existing_manifest_map() -> dict[int, dict]:
    manifest_map: dict[int, dict] = {}

    if not PREPARED_MANIFEST_PATH.exists():
        return manifest_map

    with PREPARED_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            manifest_map[int(row["track_id"])] = row

    return manifest_map


def _write_manifest_rows(rows: Iterable[dict]) -> None:
    PREPARED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREPARED_MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_data(
    bucket: str,
    prefix: str,
    only_track_ids: set[int] | None = None,
    append: bool = False,
) -> dict:
    PREPARED_TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_manifest_map = _load_existing_manifest_map() if append else {}

    all_db_tracks = load_tracks()
    s3_keys = set(list_all_songs(bucket=bucket, prefix=prefix))

    logger.info(
        "Prepare-data started bucket=%s prefix=%s s3_objects=%s db_tracks=%s append=%s",
        bucket,
        prefix,
        len(s3_keys),
        len(all_db_tracks),
        append,
    )

    invalid_items = []
    processed = 0
    skipped = 0

    candidate_tracks = []
    for row in all_db_tracks:
        track_id = int(row["track_id"])
        s3_key = row["s3_key"]

        if only_track_ids is not None and track_id not in only_track_ids:
            continue

        if s3_key not in s3_keys:
            continue

        if append and track_id in existing_manifest_map:
            continue

        candidate_tracks.append(row)

    logger.info("Prepare-data candidates=%s", len(candidate_tracks))

    for idx, row in enumerate(candidate_tracks, start=1):
        track_id = int(row["track_id"])
        key = row["s3_key"]

        tmp_path = None
        try:
            tmp_path = download_song(bucket, key)
            audio = load_audio(tmp_path)

            if audio is None or len(audio) == 0:
                raise ValueError("decoded empty audio")

            out_path = PREPARED_TRACKS_DIR / f"{track_id}.npy"
            np.save(out_path, audio.astype("float32"))

            manifest_row = {
                "track_id": track_id,
                "s3_key": key,
                "prepared_path": str(out_path),
                "num_samples": int(len(audio)),
                "duration_sec": float(len(audio) / 16000.0),
                "split": assign_split(track_id),
            }
            existing_manifest_map[track_id] = manifest_row
            processed += 1

            if idx % 100 == 0 or idx == len(candidate_tracks):
                logger.info(
                    "Prepare-data progress scanned=%s/%s processed=%s skipped=%s",
                    idx,
                    len(candidate_tracks),
                    processed,
                    skipped,
                )

        except Exception as exc:  # noqa: BLE001
            skipped += 1
            invalid_items.append({"s3_key": key, "track_id": track_id, "reason": str(exc)})
            logger.warning("Prepare-data skipped key=%s track_id=%s error=%s", key, track_id, exc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    ordered_rows = [existing_manifest_map[k] for k in sorted(existing_manifest_map.keys())]
    _write_manifest_rows(ordered_rows)

    INVALID_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INVALID_KEYS_PATH.open("w", encoding="utf-8") as f:
        json.dump(invalid_items, f, ensure_ascii=False, indent=2)

    summary = {
        "processed": processed,
        "skipped": skipped,
        "manifest_path": str(PREPARED_MANIFEST_PATH),
        "invalid_path": str(INVALID_KEYS_PATH),
        "total_prepared_tracks": len(ordered_rows),
    }

    logger.info(
        "Prepare-data finished processed=%s skipped=%s total_prepared_tracks=%s manifest=%s",
        processed,
        skipped,
        len(ordered_rows),
        PREPARED_MANIFEST_PATH,
    )
    return summary