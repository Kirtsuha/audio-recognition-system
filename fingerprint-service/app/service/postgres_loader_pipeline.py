import os
from io import BytesIO
from pathlib import PurePosixPath
import tempfile
import shutil
import pandas as pd
import boto3
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from repository.database_config import Base, engine

# ---------------- CONFIG ----------------

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio123")
S3_BUCKET = os.getenv("S3_BUCKET", "tracks")
S3_PREFIX = "fma/"

FMA_METADATA_CSV = r"/app/app/data/tracks.csv"

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg")

# ---------------- S3 INIT (MINIO) ----------------

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name="us-east-1",
    use_ssl=False
)

# ---------------- REPO / FINGERPRINT ----------------
from repository.db import get_db
from repository.models import Track, Fingerprint
from service.audio2fingerprint import fingerprint_audio

# ---------------- METADATA ----------------

df_tracks = pd.read_csv(
    FMA_METADATA_CSV,
    index_col=0,
    low_memory=False
)

# ---------------- HELPERS ----------------

def is_audio_file(path: str) -> bool:
    return path.lower().endswith(AUDIO_EXT)


def extract_track_id_from_s3_key(s3_key: str) -> int:
    filename = PurePosixPath(s3_key).name
    return int(filename.split(".")[0])


def track_already_processed(db: Session, track_id: int) -> bool:
    return db.query(Fingerprint.track_id)\
             .filter(Fingerprint.track_id == track_id)\
             .first() is not None


def upload_track_to_db(db: Session, track_id: int, s3_key: str):
    try:
        title = df_tracks.loc[track_id, ("track", "title")]
    except Exception:
        title = str(track_id)

    try:
        artist = df_tracks.loc[track_id, ("artist", "name")]
    except Exception:
        artist = "unknown"

    existing = db.query(Track).filter(Track.id == track_id).first()
    if existing:
        return existing

    track = Track(
        id=track_id,
        title=str(title),
        artist=str(artist),
        s3_key=s3_key
    )
    db.add(track)
    db.commit()
    db.refresh(track)
    return track

# ---------------- MAIN PROCESS ----------------

def process_s3_track(s3_key: str, db: Session):

    if not is_audio_file(s3_key):
        return

    track_id = extract_track_id_from_s3_key(s3_key)

    if track_already_processed(db, track_id):
        print(f"SKIP {track_id}")
        return

    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    audio_stream = BytesIO(obj["Body"].read())

    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        shutil.copyfileobj(audio_stream, tmp)
        tmp.flush()
        hashes = fingerprint_audio(tmp.name)

    track = upload_track_to_db(db, track_id, s3_key)

    # ---------------- Safe bulk insert ----------------
    batch = [
        {"hash": int(h), "track_id": track.id, "time_offset": int(t)}
        for h, t in hashes
    ]
    if batch:
        stmt = insert(Fingerprint).values(batch)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=['hash', 'track_id', 'time_offset']
        )
        db.execute(stmt)
        db.commit()
    # -------------------------------------------------

    print(f"OK {track_id} → hashes={len(hashes)}")


def process_s3_bucket():

    db: Session = next(get_db())

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            process_s3_track(obj["Key"], db)

    db.close()

# ---------------- ENTRY ----------------

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    process_s3_bucket()
