import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query, BackgroundTasks, Form
from sqlalchemy import text
from sqlalchemy.orm import Session
import threading
import uuid
from datetime import datetime, timezone
from fastapi import Query

from config.config import INDEX_DURATION_SEC, MIN_ALIGNED_MATCHES, MIN_QUERY_COVERAGE, MIN_SCORE_GAP
from fingerprint.matcher import match, retrieve_candidates
from repository.db import get_db
from repository.models import Track
from scripts.fma_to_s3_loader import upload_hf_fma_to_s3
from scripts.s3_loader import upload_single_track, upload_tracks_zip
from service.audio2fingerprint import fingerprint_audio
from service.postgres_loader_pipeline import process_s3_bucket, upload_track_to_db
from logging_utils import configure_logging
from repository.init_db import Base
from repository.database_config import engine

from fingerprint_experiment.schemas import (
    AsyncJobResponse,
    FingerprintExperimentRequest,
)
from fingerprint_experiment.runner import run_fingerprint_experiment
from fingerprint_experiment.job_status import get_status, update_status, utc_now_iso

app = FastAPI(title="Fingerprint Music Recognition service")
UPLOAD_JOBS = {}
UPLOAD_JOBS_LOCK = threading.Lock()
EXPERIMENT_IN_PROGRESS = False

configure_logging()
logger = logging.getLogger("fingerprint-service")

@app.on_event("startup")
def init_db():
    Base.metadata.create_all(engine)

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    SELECT setval(
                        pg_get_serial_sequence('track', 'id'),
                        COALESCE((SELECT MAX(id) FROM track), 0) + 1,
                        false
                    )
                    """
                )
            )
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "track_s3_key_unique_idx ON track (s3_key)"
                )
            )
            connection.execute(
                text("ALTER TABLE track ADD COLUMN IF NOT EXISTS album TEXT")
            )
    except Exception:
        logger.warning("Failed to synchronize track schema helpers", exc_info=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


def serialize_track(track: Track) -> dict:
    return {
        "track_id": track.id,
        "title": track.title,
        "artist": track.artist,
        "album": track.album,
        "s3_key": track.s3_key,
    }


def recognize_file(file: UploadFile, db: Session) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "query.wav").suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    try:
        hashes = fingerprint_audio(path, INDEX_DURATION_SEC)
        result = match(hashes, db)

        if not result.get("matched"):
            return {
                "match": False,
                "confidence": result.get("confidence", 0.0),
                "reason": result.get("reason"),
                "source": "fingerprint",
                "debug": result,
            }

        track = db.query(Track).filter(Track.id == result["track_id"]).first()

        response = {
            "match": True,
            "track_id": result["track_id"],
            "confidence": result["confidence"],
            "source": "fingerprint",
            "debug": result,
        }

        if track is not None:
            response.update(serialize_track(track))

        return response

    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/recognize")
def recognize_api(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return recognize_file(file, db)


@app.post("/index")
def legacy_index_alias(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return recognize_file(file, db)

def retrieve_candidates_file(file: UploadFile, db: Session, top_k: int = 50) -> dict:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.filename or "query.wav").suffix,
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    try:
        hashes = fingerprint_audio(path, INDEX_DURATION_SEC)
        result = retrieve_candidates(hashes, db, top_k=top_k)

        candidates = result.get("candidates") or []

        best_candidate = candidates[0] if candidates else None

        matched = False
        track_id = None
        confidence = 0.0
        reason = result.get("reason")

        if best_candidate is not None:
            track_id = best_candidate.get("track_id")
            confidence = best_candidate.get("confidence", 0.0)

            aligned_matches = int(best_candidate.get("aligned_matches") or 0)
            query_hashes = int(result.get("query_hashes") or 0)
            score_gap = float(best_candidate.get("score_gap") or 0.0)

            matched = (
                    aligned_matches >= MIN_ALIGNED_MATCHES
                    and (aligned_matches / max(1, query_hashes)) >= MIN_QUERY_COVERAGE
                    and score_gap >= MIN_SCORE_GAP
            )

            reason = None if matched else "low_confidence"

        track_ids = [
            int(candidate["track_id"])
            for candidate in result.get("candidates", [])
        ]

        tracks = (
            db.query(Track)
            .filter(Track.id.in_(track_ids))
            .all()
            if track_ids
            else []
        )

        tracks_by_id = {int(track.id): track for track in tracks}

        enriched_candidates = []

        for candidate in result.get("candidates", []):
            track = tracks_by_id.get(int(candidate["track_id"]))

            item = dict(candidate)

            if track is not None:
                item.update(
                    {
                        "title": track.title,
                        "artist": track.artist,
                        "s3_key": track.s3_key,
                    }
                )

            enriched_candidates.append(item)

        return {
            "match": matched,
            "matched": matched,
            "track_id": track_id,
            "confidence": confidence,
            "reason": reason,
            "source": "fingerprint",
            "top_k": top_k,
            "query_hashes": result.get("query_hashes", 0),
            "unique_query_hashes": result.get("unique_query_hashes", 0),
            "best_aligned_matches": result.get("best_aligned_matches", 0),
            "second_aligned_matches": result.get("second_aligned_matches", 0),
            "candidates": enriched_candidates,
        }

    finally:
        try:
            os.remove(path)
        except OSError:
            pass

@app.post("/retrieve-candidates")
def retrieve_candidates_api(
    file: UploadFile = File(...),
    top_k: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    return retrieve_candidates_file(
        file=file,
        db=db,
        top_k=top_k,
    )

@app.get("/tracks/{track_id}")
def get_track(track_id: int, db: Session = Depends(get_db)):
    track = db.query(Track).filter(Track.id == track_id).first()
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return serialize_track(track)


@app.post("/s3/upload-archive")
def upload_archive_to_s3(
    file: UploadFile = File(...),
    manifest: UploadFile = File(...),
    bucket: str | None = Form(default=None),
    prefix: str = Form(default=""),
    max_files: int | None = Form(default=None),
):
    suffix = Path(file.filename or "archive.zip").suffix.lower()
    if suffix != ".zip":
        raise HTTPException(status_code=400, detail="Only .zip archives are supported")

    manifest_suffix = Path(manifest.filename or "manifest.csv").suffix.lower()
    if manifest_suffix != ".csv":
        raise HTTPException(status_code=400, detail="Only .csv manifests are supported")

    print("Received ZIP archive upload: %s", file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        shutil.copyfileobj(file.file, tmp)
        archive_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(manifest.file, tmp)
        manifest_path = tmp.name

    try:
        summary = upload_tracks_zip(
            zip_path=archive_path,
            manifest_path=manifest_path,
            bucket=bucket,
            prefix=prefix,
            max_files=max_files,
        )
        print("ZIP archive upload completed: %s", summary)
        return summary
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            os.remove(archive_path)
        except OSError:
            print("Failed to remove temporary archive %s", archive_path)

        try:
            os.remove(manifest_path)
        except OSError:
            print("Failed to remove temporary manifest %s", manifest_path)


@app.post("/s3/upload-track")
def upload_track_to_s3(
    file: UploadFile = File(...),
    title: str = Form(...),
    artist: str = Form(...),
    album: str | None = Form(default=None),
    bucket: str | None = Form(default=None),
    prefix: str = Form(default=""),
    overwrite: bool = Form(default=False),
    db: Session = Depends(get_db),
):
    try:
        summary = upload_single_track(
            fileobj=file.file,
            filename=file.filename or "track.mp3",
            title=title,
            artist=artist,
            album=album,
            bucket=bucket,
            prefix=prefix,
            overwrite=overwrite,
        )

        track = upload_track_to_db(
            db=db,
            track_id=None,
            s3_key=summary["s3_key"],
            title=title,
            artist=artist,
            album=album,
            commit=True,
        )

        return {
            **summary,
            "track_id": int(track.id),
            "trackId": int(track.id),
            "status": "CREATED" if summary.get("uploaded") else "EXISTS",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/index/s3")
def index_s3_tracks(s3_bucket: str, s3_prefix: str | None = None):
    if s3_prefix is None:
        s3_prefix = ""
    print("Received request to index tracks from S3 into Postgres")
    summary = process_s3_bucket(s3_bucket, s3_prefix)
    print("S3 indexing request completed: %s", summary)
    return summary

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_upload_job(job_id: str, payload: dict):
    with UPLOAD_JOBS_LOCK:
        current = UPLOAD_JOBS.get(job_id, {})
        current.update(payload)
        current["updated_at"] = utc_now_iso()
        UPLOAD_JOBS[job_id] = current


def run_hf_upload_job(job_id: str, max_files: int):
    update_upload_job(job_id, {
        "job_id": job_id,
        "status": "running",
        "max_files": max_files,
        "started_at": utc_now_iso(),
    })

    try:
        summary = upload_hf_fma_to_s3(
            max_files=max_files,
            job_id=job_id,
            progress_callback=lambda payload: update_upload_job(job_id, payload),
        )

        update_upload_job(job_id, {
            **summary,
            "status": "completed",
            "finished_at": utc_now_iso(),
        })

    except Exception as exc:
        update_upload_job(job_id, {
            "status": "failed",
            "error": str(exc),
            "finished_at": utc_now_iso(),
        })

@app.post("/s3/upload-hf/start")
def start_hf_upload(max_files: int = Query(8000, ge=1, le=20000)):
    job_id = str(uuid.uuid4())

    with UPLOAD_JOBS_LOCK:
        UPLOAD_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "max_files": max_files,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "uploaded": 0,
            "skipped": 0,
            "failed": 0,
            "scanned": 0,
        }

    thread = threading.Thread(
        target=run_hf_upload_job,
        args=(job_id, max_files),
        daemon=True,
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "max_files": max_files,
        "status_url": f"/s3/upload-hf/status/{job_id}",
    }


@app.get("/s3/upload-hf/status/{job_id}")
def get_hf_upload_status(job_id: str):
    with UPLOAD_JOBS_LOCK:
        job = UPLOAD_JOBS.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Upload job not found")

    return job


@app.get("/s3/upload-hf/status")
def list_hf_upload_jobs():
    with UPLOAD_JOBS_LOCK:
        return {
            "jobs": list(UPLOAD_JOBS.values())
        }

def run_fingerprint_experiment_job(payload: FingerprintExperimentRequest):
    global EXPERIMENT_IN_PROGRESS

    if EXPERIMENT_IN_PROGRESS:
        return

    EXPERIMENT_IN_PROGRESS = True

    update_status(
        current_job="fingerprint-experiment",
        phase="running",
        started_at=utc_now_iso(),
        finished_at=None,
        last_error=None,
        progress={
            "experiment_name": payload.experiment_name,
            "bucket": payload.bucket,
            "prefix": payload.prefix,
            "track_limit": payload.track_limit,
            "query_limit": payload.query_limit,
        },
    )

    try:
        summary = run_fingerprint_experiment(payload)

        update_status(
            phase="done",
            finished_at=utc_now_iso(),
            last_success=utc_now_iso(),
            progress={
                "summary": summary,
            },
        )

    except Exception as exc:
        logger.exception("Fingerprint experiment failed")

        update_status(
            phase="failed",
            finished_at=utc_now_iso(),
            last_error=str(exc),
        )

    finally:
        EXPERIMENT_IN_PROGRESS = False


@app.post("/experiments/fingerprint/run", response_model=AsyncJobResponse)
def fingerprint_experiment_run(
    payload: FingerprintExperimentRequest,
    background_tasks: BackgroundTasks,
):
    if EXPERIMENT_IN_PROGRESS:
        raise HTTPException(status_code=409, detail="Fingerprint experiment is already running")

    background_tasks.add_task(run_fingerprint_experiment_job, payload)

    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        run_dir=None,
    )


@app.post("/experiments/fingerprint/run-sync")
def fingerprint_experiment_run_sync(payload: FingerprintExperimentRequest):
    if EXPERIMENT_IN_PROGRESS:
        raise HTTPException(status_code=409, detail="Fingerprint experiment is already running")

    return run_fingerprint_experiment(payload)


@app.get("/experiments/fingerprint/status")
def fingerprint_experiment_status():
    return {
        "experiment_in_progress": EXPERIMENT_IN_PROGRESS,
        "job_status": get_status(),
    }
