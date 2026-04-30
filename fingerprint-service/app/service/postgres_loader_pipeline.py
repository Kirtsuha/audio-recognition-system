import logging
import os
import shutil
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import PurePosixPath

import pandas as pd
import soundfile as sf
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from config.config import INDEX_DURATION_SEC
from repository.database_config import Base, engine
from repository.db import get_db
from repository.models import Fingerprint, Track
from repository.s3_client import get_s3
from service.audio2fingerprint import fingerprint_audio

logger = logging.getLogger("fingerprint.indexer")

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg")

INDEX_WORKERS = int(os.getenv("FP_INDEX_WORKERS", "1"))
INSERT_BATCH_SIZE = int(os.getenv("FP_INSERT_BATCH_SIZE", "10000"))
FAST_REINDEX_MODE = os.getenv("FP_FAST_REINDEX_MODE", "false").lower() == "true"
PROGRESS_EVERY = int(os.getenv("FP_INDEX_PROGRESS_EVERY", "25"))
MAX_TRACKS = int(os.getenv("FP_INDEX_MAX_TRACKS", "0"))

s3 = get_s3()
df_tracks = None


def get_tracks_metadata():
    global df_tracks

    if df_tracks is None:
        df_tracks = pd.DataFrame()

    return df_tracks


def is_audio_file(path: str) -> bool:
    return path.lower().endswith(AUDIO_EXT)


def extract_track_id_from_s3_key(s3_key: str) -> int:
    filename = PurePosixPath(s3_key).name
    stem = filename.split(".")[0]

    if stem.isdigit():
        return int(stem)

    return zlib.crc32(s3_key.encode("utf-8")) & 0x7FFFFFFF


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def get_audio_duration_sec(path: str) -> float | None:
    try:
        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames / info.samplerate)
    except Exception:
        logger.warning("Failed to read duration using soundfile: %s", path, exc_info=True)

    return None


def track_already_processed(db: Session, track_id: int) -> bool:
    return (
        db.query(Fingerprint.track_id)
        .filter(Fingerprint.track_id == track_id)
        .first()
        is not None
    )


def upload_track_to_db(
    db: Session,
    track_id: int,
    s3_key: str,
    duration_sec: float | None = None,
    fingerprint_count: int | None = None,
    commit: bool = True,
) -> Track:
    path = PurePosixPath(s3_key)
    default_title = path.stem
    default_artist = path.parent.name if path.parent.name else "unknown"

    metadata = get_tracks_metadata()

    try:
        title = metadata.loc[track_id, ("track", "title")]
    except Exception:
        title = default_title

    try:
        artist = metadata.loc[track_id, ("artist", "name")]
    except Exception:
        artist = default_artist

    existing = db.query(Track).filter(Track.id == track_id).first()

    if existing:
        existing.duration_sec = (
            int(duration_sec) if duration_sec is not None else existing.duration_sec
        )
        existing.fingerprint_count = (
            fingerprint_count if fingerprint_count is not None else existing.fingerprint_count
        )

        if commit:
            db.commit()
            db.refresh(existing)

        return existing

    track = Track(
        id=track_id,
        title=str(title),
        artist=str(artist),
        s3_key=s3_key,
        duration_sec=int(duration_sec) if duration_sec is not None else None,
        fingerprint_count=fingerprint_count,
    )

    db.add(track)

    if commit:
        db.commit()
        db.refresh(track)
    else:
        db.flush()

    return track


def _fingerprint_s3_track_worker(
    s3_key: str,
    s3_bucket: str,
) -> dict:
    """
    Worker process:
    - creates its own S3 client
    - downloads track
    - calculates duration
    - calculates fingerprint hashes
    - returns pure dict, no DB objects
    """
    worker_s3 = get_s3()
    track_id = extract_track_id_from_s3_key(s3_key)

    timings = {}
    started = time.perf_counter()

    obj = worker_s3.get_object(Bucket=s3_bucket, Key=s3_key)
    audio_stream = BytesIO(obj["Body"].read())
    timings["download_sec"] = time.perf_counter() - started

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        shutil.copyfileobj(audio_stream, tmp)
        tmp.flush()

        t_duration = time.perf_counter()
        duration_sec = get_audio_duration_sec(tmp.name)
        timings["duration_sec_read_sec"] = time.perf_counter() - t_duration

        t_fp = time.perf_counter()
        hashes = fingerprint_audio(tmp.name, duration=INDEX_DURATION_SEC)
        timings["fingerprint_sec"] = time.perf_counter() - t_fp

    timings["total_worker_sec"] = time.perf_counter() - started

    return {
        "status": "fingerprinted",
        "track_id": track_id,
        "s3_key": s3_key,
        "duration_sec": duration_sec,
        "hashes": [(int(h), int(t)) for h, t in hashes],
        "hash_count": len(hashes),
        "hashes_per_second": (
            len(hashes) / max(1.0, duration_sec)
            if duration_sec is not None
            else None
        ),
        "timings": timings,
    }


def insert_fingerprint_result(db: Session, result: dict) -> dict:
    """
    Main process only:
    - writes Track
    - writes Fingerprint rows
    """
    t0 = time.perf_counter()

    track = upload_track_to_db(
        db=db,
        track_id=int(result["track_id"]),
        s3_key=result["s3_key"],
        duration_sec=result.get("duration_sec"),
        fingerprint_count=int(result.get("hash_count") or 0),
        commit=False,
    )

    t_track = time.perf_counter()

    batch = [
        {
            "hash": int(hash_value),
            "track_id": int(track.id),
            "time_offset": int(time_offset),
        }
        for hash_value, time_offset in result["hashes"]
    ]

    if batch:
        for part in chunks(batch, INSERT_BATCH_SIZE):
            stmt = insert(Fingerprint).values(part)

            if not FAST_REINDEX_MODE:
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["hash", "track_id", "time_offset"]
                )

            db.execute(stmt)

    db.commit()

    t_insert = time.perf_counter()

    result["timings"]["db_track_sec"] = t_track - t0
    result["timings"]["db_insert_sec"] = t_insert - t_track
    result["timings"]["db_total_sec"] = t_insert - t0

    return {
        "status": "indexed",
        "track_id": int(result["track_id"]),
        "s3_key": result["s3_key"],
        "duration_sec": result.get("duration_sec"),
        "hashes": int(result.get("hash_count") or 0),
        "hashes_per_second": result.get("hashes_per_second"),
        "timings": result["timings"],
    }

def load_existing_track_ids(db: Session) -> set[int]:
    started = time.perf_counter()
    rows = db.query(Track.id).all()
    ids = {int(row[0]) for row in rows}

    logger.info(
        "Loaded existing track ids count=%s elapsed_sec=%.2f",
        len(ids),
        time.perf_counter() - started,
    )

    return ids


def process_s3_track(s3_key: str, db: Session, s3_bucket: str, s3_prefix: str = "") -> dict:
    """
    Sequential mode. Useful for debugging and INDEX_WORKERS=1.
    """
    if not is_audio_file(s3_key):
        logger.info("Skipping non-audio object: %s", s3_key)
        return {"status": "ignored", "s3_key": s3_key}

    track_id = extract_track_id_from_s3_key(s3_key)

    if not FAST_REINDEX_MODE and track_already_processed(db, track_id):
        logger.info("Skipping already indexed track %s from %s", track_id, s3_key)
        return {"status": "skipped", "track_id": track_id, "s3_key": s3_key}

    logger.info("Processing track %s from S3 key %s", track_id, s3_key)

    result = _fingerprint_s3_track_worker(
        s3_key=s3_key,
        s3_bucket=s3_bucket,
    )

    indexed = insert_fingerprint_result(db, result)

    timings = indexed["timings"]

    logger.info(
        "Indexed track %s: hashes=%s duration_sec=%s hashes_per_second=%s "
        "download=%.3f duration_read=%.3f fingerprint=%.3f db_insert=%.3f total=%.3f",
        indexed["track_id"],
        indexed["hashes"],
        indexed["duration_sec"],
        indexed["hashes_per_second"],
        timings.get("download_sec", 0.0),
        timings.get("duration_sec_read_sec", 0.0),
        timings.get("fingerprint_sec", 0.0),
        timings.get("db_insert_sec", 0.0),
        timings.get("total_worker_sec", 0.0) + timings.get("db_total_sec", 0.0),
    )

    return indexed


def _list_s3_audio_keys(s3_bucket: str, s3_prefix: str) -> list[str]:
    keys = []

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if is_audio_file(key):
                keys.append(key)

    keys.sort()

    if MAX_TRACKS > 0:
        keys = keys[:MAX_TRACKS]

    return keys


def _process_s3_bucket_sequential(s3_bucket: str, s3_prefix: str, keys: list[str]) -> dict:
    stats = {
        "indexed": 0,
        "skipped": 0,
        "ignored": 0,
        "failed": 0,
        "processed_keys": 0,
        "total_keys": len(keys),
        "total_hashes": 0,
        "elapsed_sec": 0.0,
    }

    started = time.perf_counter()
    db: Session = next(get_db())

    try:
        for key in keys:
            stats["processed_keys"] += 1

            try:
                result = process_s3_track(key, db, s3_bucket, s3_prefix)
                status = result["status"]

                if status == "indexed":
                    stats["indexed"] += 1
                    stats["total_hashes"] += int(result.get("hashes") or 0)
                elif status == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["ignored"] += 1

            except Exception as exc:
                db.rollback()
                stats["failed"] += 1
                logger.exception("Failed to process key=%s error=%s", key, exc)

            if stats["processed_keys"] % PROGRESS_EVERY == 0 or stats["processed_keys"] == len(keys):
                elapsed = time.perf_counter() - started
                tracks_per_sec = stats["processed_keys"] / max(1e-6, elapsed)

                logger.info(
                    "Index progress processed=%s/%s indexed=%s skipped=%s failed=%s "
                    "total_hashes=%s elapsed_sec=%.1f tracks_per_sec=%.3f",
                    stats["processed_keys"],
                    stats["total_keys"],
                    stats["indexed"],
                    stats["skipped"],
                    stats["failed"],
                    stats["total_hashes"],
                    elapsed,
                    tracks_per_sec,
                )

        stats["elapsed_sec"] = time.perf_counter() - started
        return stats

    finally:
        db.close()


def _process_s3_bucket_parallel(s3_bucket: str, s3_prefix: str, keys: list[str]) -> dict:
    stats = {
        "indexed": 0,
        "skipped": 0,
        "ignored": 0,
        "failed": 0,
        "processed_keys": 0,
        "total_keys": len(keys),
        "total_hashes": 0,
        "elapsed_sec": 0.0,
    }

    started = time.perf_counter()

    db: Session = next(get_db())

    try:
        keys_to_process = []

        if FAST_REINDEX_MODE:
            keys_to_process = keys
        else:
            existing_track_ids = load_existing_track_ids(db)

            for idx, key in enumerate(keys, start=1):
                track_id = extract_track_id_from_s3_key(key)

                if track_id in existing_track_ids:
                    stats["skipped"] += 1
                    stats["processed_keys"] += 1
                else:
                    keys_to_process.append(key)

                if idx % PROGRESS_EVERY == 0 or idx == len(keys):
                    logger.info(
                        "Precheck progress checked=%s/%s skipped_existing=%s to_process=%s",
                        idx,
                        len(keys),
                        stats["skipped"],
                        len(keys_to_process),
                    )

        logger.info(
            "Parallel indexing started workers=%s keys=%s skipped_existing=%s fast_reindex=%s",
            INDEX_WORKERS,
            len(keys_to_process),
            stats["skipped"],
            FAST_REINDEX_MODE,
        )

        with ProcessPoolExecutor(max_workers=INDEX_WORKERS) as executor:
            futures = {
                executor.submit(_fingerprint_s3_track_worker, key, s3_bucket): key
                for key in keys_to_process
            }

            for future in as_completed(futures):
                key = futures[future]
                stats["processed_keys"] += 1

                try:
                    result = future.result()
                    indexed = insert_fingerprint_result(db, result)

                    stats["indexed"] += 1
                    stats["total_hashes"] += int(indexed.get("hashes") or 0)

                    timings = indexed["timings"]

                    logger.info(
                        "Indexed track %s: hashes=%s duration_sec=%s hashes_per_second=%s "
                        "download=%.3f duration_read=%.3f fingerprint=%.3f db_insert=%.3f worker_total=%.3f",
                        indexed["track_id"],
                        indexed["hashes"],
                        indexed["duration_sec"],
                        indexed["hashes_per_second"],
                        timings.get("download_sec", 0.0),
                        timings.get("duration_sec_read_sec", 0.0),
                        timings.get("fingerprint_sec", 0.0),
                        timings.get("db_insert_sec", 0.0),
                        timings.get("total_worker_sec", 0.0),
                    )

                except Exception as exc:
                    db.rollback()
                    stats["failed"] += 1
                    logger.exception("Failed to process key=%s error=%s", key, exc)

                if stats["processed_keys"] % PROGRESS_EVERY == 0 or stats["processed_keys"] == stats["total_keys"]:
                    elapsed = time.perf_counter() - started
                    tracks_per_sec = stats["processed_keys"] / max(1e-6, elapsed)

                    logger.info(
                        "Index progress processed=%s/%s indexed=%s skipped=%s failed=%s "
                        "total_hashes=%s elapsed_sec=%.1f tracks_per_sec=%.3f",
                        stats["processed_keys"],
                        stats["total_keys"],
                        stats["indexed"],
                        stats["skipped"],
                        stats["failed"],
                        stats["total_hashes"],
                        elapsed,
                        tracks_per_sec,
                    )

        stats["elapsed_sec"] = time.perf_counter() - started
        return stats

    finally:
        db.close()


def process_s3_bucket(s3_bucket: str, s3_prefix: str) -> dict:
    logger.info(
        "S3 indexing requested bucket=%s prefix=%s workers=%s fast_reindex=%s "
        "insert_batch_size=%s max_tracks=%s",
        s3_bucket,
        s3_prefix,
        INDEX_WORKERS,
        FAST_REINDEX_MODE,
        INSERT_BATCH_SIZE,
        MAX_TRACKS or None,
    )

    started = time.perf_counter()

    keys = _list_s3_audio_keys(s3_bucket, s3_prefix)

    logger.info(
        "S3 listing finished bucket=%s prefix=%s audio_keys=%s elapsed_sec=%.2f",
        s3_bucket,
        s3_prefix,
        len(keys),
        time.perf_counter() - started,
    )

    if INDEX_WORKERS <= 1:
        stats = _process_s3_bucket_sequential(s3_bucket, s3_prefix, keys)
    else:
        stats = _process_s3_bucket_parallel(s3_bucket, s3_prefix, keys)

    logger.info("S3 indexing finished stats=%s", stats)

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    Base.metadata.create_all(bind=engine)

    bucket = os.getenv("S3_BUCKET", "tracks")
    prefix = os.getenv("S3_PREFIX", "")

    process_s3_bucket(bucket, prefix)