

from fastapi import FastAPI, UploadFile, File, Depends
import tempfile
import shutil

from sqlalchemy.orm import Session

from fingerprint.matcher import match
from repository.db import get_db
from service.audio2fingerprint import fingerprint_audio

from repository.init_db import Base
from repository.database_config import engine
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Music Recognition service")

@app.get("/health")
async def health():
    return {"status": "ok"}

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

    return {
        "match": True,
        "track_id": result["track_id"],
        "confidence": min(1.0, result["matches"] / 100)
    }