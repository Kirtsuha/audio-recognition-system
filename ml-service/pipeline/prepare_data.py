import json
import logging
import os

import numpy as np

from db.track_repo import load_tracks
from pipeline.config import INVALID_KEYS_PATH, PREPARED_TRACKS_DIR
from pipeline.dataset import assign_split, load_audio
from pipeline.manifest import load_prepared_manifest_map, write_prepared_manifest
from s3.s3_list import list_all_songs
from s3.s3_loader import download_song

logger = logging.getLogger("ml-pipeline.prepare-data")


def prepare_data(
    bucket: str,
    prefix: str,
    only_track_ids: set[int] | None = None,
    append: bool = False,
    prepare_limit: int = 0,
) -> dict:
    PREPARED_TRACKS_DIR.mkdir(parents=True, exist_ok=True)

    existing_manifest_map = load_prepared_manifest_map() if append else {}

    db_tracks = load_tracks()
    s3_keys = set(list_all_songs(bucket=bucket, prefix=prefix))

    candidate_tracks = []

    for row in db_tracks:
        track_id = int(row["track_id"])
        s3_key = row["s3_key"]

        if only_track_ids is not None and track_id not in only_track_ids:
            continue

        if s3_key not in s3_keys:
            continue

        if append and track_id in existing_manifest_map:
            continue

        candidate_tracks.append(row)

    candidate_tracks.sort(key=lambda x: int(x["track_id"]))

    if prepare_limit > 0:
        candidate_tracks = candidate_tracks[:prepare_limit]

    logger.info(
        "Prepare-data started bucket=%s prefix=%s s3_objects=%s db_tracks=%s append=%s candidates=%s",
        bucket,
        prefix,
        len(s3_keys),
        len(db_tracks),
        append,
        len(candidate_tracks),
    )

    invalid_items = []
    processed = 0
    skipped = 0

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

            existing_manifest_map[track_id] = {
                "track_id": track_id,
                "s3_key": key,
                "prepared_path": str(out_path),
                "num_samples": int(len(audio)),
                "duration_sec": float(len(audio) / 16000.0),
                "split": assign_split(track_id),
            }

            processed += 1

            if idx % 100 == 0 or idx == len(candidate_tracks):
                logger.info(
                    "Prepare-data progress scanned=%s/%s processed=%s skipped=%s",
                    idx,
                    len(candidate_tracks),
                    processed,
                    skipped,
                )

        except Exception as exc:
            skipped += 1
            invalid_items.append({"s3_key": key, "track_id": track_id, "reason": str(exc)})
            logger.warning("Prepare-data skipped key=%s track_id=%s error=%s", key, track_id, exc)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    write_prepared_manifest(existing_manifest_map.values())

    INVALID_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INVALID_KEYS_PATH.open("w", encoding="utf-8") as f:
        json.dump(invalid_items, f, ensure_ascii=False, indent=2)

    summary = {
        "processed": processed,
        "skipped": skipped,
        "invalid_path": str(INVALID_KEYS_PATH),
        "total_prepared_tracks": len(existing_manifest_map),
        "prepare_limit": prepare_limit,
    }

    logger.info("Prepare-data finished summary=%s", summary)
    return summary