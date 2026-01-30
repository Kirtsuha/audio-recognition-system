

from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil

from fingerprint.matcher import match
from service.audio2fingerprint import fingerprint_audio

app = FastAPI(title="Music Recognition service")

fingerprint_db = {}
track_meta = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/index")
def index_track_api(
        track_id: int,
        title: str,
        artist: str,
        file: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    hashes = fingerprint_audio(path)
    result = match(hashes, fingerprint_db)
    print(hashes, result)

    if not result:
        return {"match": False}

    track = track_meta[result["track_id"]]

    return {
        "match": True,
        "track": track,
        "confidence": min(1.0, result["matches"] / 100)
    }


@app.get("/")
def root():
    return {"message": "Hello World"}