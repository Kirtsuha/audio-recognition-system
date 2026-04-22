import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from fingerprint.matcher import match
from repository.db import get_db
from repository.models import Track
from scripts.s3_loader import upload_fma_zip
from service.audio2fingerprint import fingerprint_audio
from service.postgres_loader_pipeline import process_s3_bucket

from repository.init_db import Base
from repository.database_config import engine

app = FastAPI(title="Fingerprint Music Recognition service")


@app.on_event("startup")
def init_db():
    Base.metadata.create_all(engine)


@app.get("/health")
async def health():
    return {"status": "ok"}


def serialize_track(track: Track) -> dict:
    return {
        "track_id": track.id,
        "title": track.title,
        "artist": track.artist,
        "s3_key": track.s3_key,
    }


@app.post("/index")
def index_track_api(
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    print("Recognition request started for file=%s", file.filename)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    try:
        file_size = os.path.getsize(path)
        print("Temporary audio file saved to %s (%s bytes)", path, file_size)

        hashes = fingerprint_audio(path)
        print("Generated %s hashes for request file=%s", len(hashes), file.filename)

        result = match(hashes, db)

        if not result:
            print("No match found for file=%s", file.filename)
            return {"match": False}

        track = db.query(Track).filter(Track.id == result["track_id"]).first()
        if track is None:
            response = {
                "match": True,
                "track_id": result["track_id"],
                "confidence": min(1.0, result["matches"] / 100)
            }
            print("Match found without metadata: %s", response)
            return response

        response = {
            "match": True,
            **serialize_track(track),
            "confidence": min(1.0, result["matches"] / 100)
        }
        print("Match found for file=%s: track_id=%s", file.filename, track.id)
        return response
    finally:
        try:
            os.remove(path)
        except OSError:
            print("Failed to remove temporary file %s", path)


@app.get("/tracks/{track_id}")
def get_track(track_id: int, db: Session = Depends(get_db)):
    track = db.query(Track).filter(Track.id == track_id).first()
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return serialize_track(track)


@app.post("/s3/upload-archive")
def upload_archive_to_s3(file: UploadFile = File(...)):
    suffix = Path(file.filename or "archive.zip").suffix.lower()
    if suffix != ".zip":
        raise HTTPException(status_code=400, detail="Only .zip archives are supported")

    print("Received ZIP archive upload: %s", file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        shutil.copyfileobj(file.file, tmp)
        archive_path = tmp.name

    try:
        summary = upload_fma_zip(archive_path)
        print("ZIP archive upload completed: %s", summary)
        return summary
    finally:
        try:
            os.remove(archive_path)
        except OSError:
            print("Failed to remove temporary archive %s", archive_path)


@app.post("/index/s3")
def index_s3_tracks():
    print("Received request to index tracks from S3 into Postgres")
    summary = process_s3_bucket()
    print("S3 indexing request completed: %s", summary)
    return summary
