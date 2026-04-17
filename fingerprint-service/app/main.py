from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
import tempfile
import shutil

from sqlalchemy.orm import Session

from fingerprint.matcher import match
from repository.db import get_db
from repository.models import Track
from service.audio2fingerprint import fingerprint_audio

from repository.init_db import Base
from repository.database_config import engine

app = FastAPI(title="Music Recognition service")


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
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    hashes = fingerprint_audio(path)

    result = match(hashes, db)

    if not result:
        return {"match": False}

    track = db.query(Track).filter(Track.id == result["track_id"]).first()
    if track is None:
        return {
            "match": True,
            "track_id": result["track_id"],
            "confidence": min(1.0, result["matches"] / 100)
        }

    return {
        "match": True,
        **serialize_track(track),
        "confidence": min(1.0, result["matches"] / 100)
    }


@app.get("/tracks/{track_id}")
def get_track(track_id: int, db: Session = Depends(get_db)):
    track = db.query(Track).filter(Track.id == track_id).first()
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return serialize_track(track)
