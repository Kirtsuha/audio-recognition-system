import logging
import os
import shutil
import tempfile
import zlib
from io import BytesIO
from pathlib import Path, PurePosixPath
import soundfile as sf

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from repository.database_config import Base, engine
from repository.db import get_db
from repository.models import Fingerprint, Track
from repository.s3_client import get_s3
from service.audio2fingerprint import fingerprint_audio

import logging

logger = logging.getLogger("fingerprint.indexer")
AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg")

from config.config import INDEX_DURATION_SEC
s3 = get_s3()
df_tracks = None


def get_tracks_metadata():
    global df_tracks
    df_tracks = pd.DataFrame()


def is_audio_file(path: str) -> bool:
    return path.lower().endswith(AUDIO_EXT)


def extract_track_id_from_s3_key(s3_key: str) -> int:
    filename = PurePosixPath(s3_key).name
    stem = filename.split(".")[0]
    if stem.isdigit():
        return int(stem)
    return zlib.crc32(s3_key.encode("utf-8")) & 0x7FFFFFFF


def track_already_processed(db: Session, track_id: int) -> bool:
    return db.query(Fingerprint.track_id).filter(Fingerprint.track_id == track_id).first() is not None


def upload_track_to_db(
    db: Session,
    track_id: int,
    s3_key: str,
    duration_sec: float | None = None,
    fingerprint_count: int | None = None,
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
        existing.duration_sec = int(duration_sec) if duration_sec is not None else existing.duration_sec
        existing.fingerprint_count = fingerprint_count if fingerprint_count is not None else existing.fingerprint_count
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
    db.commit()
    db.refresh(track)

    return track

def get_audio_duration_sec(path: str) -> float | None:
    try:
        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames / info.samplerate)
    except Exception:
        logger.warning("Failed to read duration using soundfile: %s", path, exc_info=True)

    return None

def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def process_s3_track(s3_key: str, db: Session, s3_bucket: str, s3_prefix: str) -> dict:
    if not is_audio_file(s3_key):
        logger.info("Skipping non-audio object: %s", s3_key)
        return {"status": "ignored", "s3_key": s3_key}

    track_id = extract_track_id_from_s3_key(s3_key)

    if track_already_processed(db, track_id):
        logger.info("Skipping already indexed track %s from %s", track_id, s3_key)
        return {"status": "skipped", "track_id": track_id, "s3_key": s3_key}

    logger.info("Downloading track %s from S3 key %s", track_id, s3_key)
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    audio_stream = BytesIO(obj["Body"].read())

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        shutil.copyfileobj(audio_stream, tmp)
        tmp.flush()

        duration_sec = get_audio_duration_sec(tmp.name)

        logger.info("Generating hashes for track %s", track_id)
        hashes = fingerprint_audio(tmp.name, duration=INDEX_DURATION_SEC)

    track = upload_track_to_db(
        db=db,
        track_id=track_id,
        s3_key=s3_key,
        duration_sec=duration_sec,
        fingerprint_count=len(hashes),
    )

    batch = [
        {
            "hash": int(hash_value),
            "track_id": track.id,
            "time_offset": int(time_offset),
        }
        for hash_value, time_offset in hashes
    ]

    if batch:
        for part in chunks(batch, 10_000):
            stmt = insert(Fingerprint).values(part)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["hash", "track_id", "time_offset"]
            )
            db.execute(stmt)

        db.commit()

    hashes_per_second = (
        len(hashes) / max(1.0, duration_sec)
        if duration_sec is not None
        else None
    )

    logger.info(
        "Indexed track %s: hashes=%s duration_sec=%s hashes_per_second=%s",
        track_id,
        len(hashes),
        duration_sec,
        hashes_per_second,
    )

    return {
        "status": "indexed",
        "track_id": track_id,
        "s3_key": s3_key,
        "duration_sec": duration_sec,
        "hashes": len(hashes),
        "hashes_per_second": hashes_per_second,
    }


def process_s3_bucket(s3_bucket: str, s3_prefix: str) -> dict:
    stats = {
        "indexed": 0,
        "skipped": 0,
        "ignored": 0,
        "failed": 0,
        "processed_keys": 0,
    }

    db: Session = next(get_db())

    try:
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                stats["processed_keys"] += 1

                try:
                    result = process_s3_track(key, db, s3_bucket, s3_prefix)
                    status = result["status"]

                    if status == "indexed":
                        stats["indexed"] += 1
                    elif status == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["ignored"] += 1

                except Exception as e:
                    db.rollback()
                    stats["failed"] += 1
                    print(f"Failed to process {key}: {e}")

        return stats

    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    Base.metadata.create_all(bind=engine)
    process_s3_bucket()
