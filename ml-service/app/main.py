import atexit
import base64
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any

import faiss
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from kafka import KafkaConsumer, KafkaProducer

from app.inference import aggregate_results
from app.model import AudioEncoder
from app.schemas import AsyncJobResponse, RecognitionResponse, TrainFromS3Request
from pipeline.build_index import build_faiss_index
from pipeline.build_embedings_from_s3 import build_embeddings_from_s3
from app.logging_utils import configure_logging, log_stage

from pipeline.config import (
    MODEL_PATH,
    SONG_IDS_PATH,
    FAISS_INDEX_PATH,
    FAISS_TOP_K,
    MIN_CONFIDENCE,
    MIN_MARGIN,
    MIN_SUPPORTED_WINDOWS,
)
from pipeline.dataset import extract_sliding_windows, load_audio
from pipeline.to_mel import to_mel
from pipeline.train import train as train_from_s3_impl

app = FastAPI(title="ML Music Recognition fallback")
configure_logging()
logger = logging.getLogger("ml-service")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
ML_REQUEST_TOPIC = os.getenv("ML_REQUEST_TOPIC", "ml-recognition-requests")
ML_RESPONSE_TOPIC = os.getenv("ML_RESPONSE_TOPIC", "ml-recognition-responses")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "ml-service")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model: AudioEncoder | None = None
index: faiss.Index | None = None
song_ids: np.ndarray | None = None
producer: KafkaProducer | None = None

train_lock = threading.Lock()
build_lock = threading.Lock()
bootstrap_lock = threading.Lock()
runtime_lock = threading.Lock()

train_in_progress = False
build_in_progress = False
bootstrap_in_progress = False

job_status_lock = threading.Lock()

job_status = {
    "current_job": None,
    "phase": None,
    "started_at": None,
    "finished_at": None,
    "last_error": None,
    "last_success": None,
    "progress": {},
}

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_job_status(**fields) -> None:
    with job_status_lock:
        job_status.update(fields)


def update_job_progress(**fields) -> None:
    with job_status_lock:
        progress = dict(job_status.get("progress", {}))
        progress.update(fields)
        job_status["progress"] = progress


def reset_job_progress() -> None:
    with job_status_lock:
        job_status["progress"] = {}

def load_runtime_artifacts() -> None:
    global model, index, song_ids

    with runtime_lock:
        started = time.time()
        missing = []

        if not os.path.exists(MODEL_PATH):
            missing.append(str(MODEL_PATH))
        if not os.path.exists(FAISS_INDEX_PATH):
            missing.append(str(FAISS_INDEX_PATH))
        if not os.path.exists(SONG_IDS_PATH):
            missing.append(str(SONG_IDS_PATH))

        if missing:
            logger.warning("Runtime artifacts missing, degraded mode enabled: %s", missing)
            model = None
            index = None
            song_ids = None
            return

        logger.info("Loading runtime artifacts model=%s index=%s song_ids=%s", MODEL_PATH, FAISS_INDEX_PATH, SONG_IDS_PATH)

        loaded_model = AudioEncoder().to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        loaded_model.load_state_dict(state)
        loaded_model.eval()

        loaded_index = faiss.read_index(str(FAISS_INDEX_PATH))
        loaded_song_ids = np.load(SONG_IDS_PATH)

        if loaded_index.ntotal != len(loaded_song_ids):
            raise RuntimeError(
                f"Inconsistent runtime artifacts: index.ntotal={loaded_index.ntotal}, "
                f"song_ids={len(loaded_song_ids)}"
            )

        model = loaded_model
        index = loaded_index
        song_ids = loaded_song_ids

        elapsed = time.time() - started
        logger.info(
            "Runtime loaded successfully elapsed_sec=%.2f device=%s index_vectors=%s index_dim=%s",
            elapsed,
            device,
            loaded_index.ntotal,
            loaded_index.d,
        )


def ensure_runtime_ready() -> None:
    if model is None or index is None or song_ids is None:
        raise RuntimeError(
            "ML runtime is not ready. "
            "Artifacts are missing or not loaded. "
            "Run /admin/bootstrap-from-s3 or /admin/reload-runtime after building artifacts."
        )


def should_accept(result: dict[str, Any]) -> bool:
    return (
        result["confidence"] >= MIN_CONFIDENCE
        and result["margin"] >= MIN_MARGIN
        and result["support"] >= MIN_SUPPORTED_WINDOWS
    )


def embed_windows(windows: list[np.ndarray]) -> np.ndarray:
    ensure_runtime_ready()
    assert model is not None

    batch = torch.stack([to_mel(w) for w in windows]).to(device)

    with torch.inference_mode():
        embs = model(batch).detach().cpu().numpy().astype("float32")

    faiss.normalize_L2(embs)
    return embs


def recognize_audio_bytes(audio_bytes: bytes) -> dict[str, Any]:
    ensure_runtime_ready()
    assert index is not None
    assert song_ids is not None

    started = time.time()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    try:
        audio = load_audio(temp_path)
        windows = extract_sliding_windows(audio)

        if not windows:
            raise ValueError("Could not extract any audio windows")

        embs = embed_windows(windows)
        scores, nearest = index.search(embs, k=FAISS_TOP_K)

        result = aggregate_results(scores=scores, indices=nearest, song_ids=song_ids)
        if result is None:
            logger.info("Recognition finished matched=false reason=no_candidates windows=%s elapsed_sec=%.2f", len(windows), time.time() - started)
            return {
                "matched": False,
                "reason": "no_candidates",
                "top_candidates": [],
            }

        accepted = should_accept(result)
        result["matched"] = accepted

        if not accepted:
            result["reason"] = "low_confidence"

        logger.info(
            "Recognition finished matched=%s song_id=%s confidence=%.4f margin=%.4f support=%s windows=%s elapsed_sec=%.2f",
            accepted,
            result.get("song_id"),
            result.get("confidence", 0.0),
            result.get("margin", 0.0),
            result.get("support"),
            len(windows),
            time.time() - started,
        )

        return result

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def get_producer() -> KafkaProducer:
    global producer
    if producer is None:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            linger_ms=20,
        )
    return producer


def consume_requests_forever() -> None:
    while True:
        try:
            consumer = KafkaConsumer(
                ML_REQUEST_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=KAFKA_CONSUMER_GROUP,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            )
            logger.info("Kafka consumer started for topic %s", ML_REQUEST_TOPIC)

            for message in consumer:
                payload = message.value
                request_id = payload.get("requestId")
                audio_base64 = payload.get("audioBase64")

                if not request_id or not audio_base64:
                    logger.warning("Skipping malformed ML message: %s", payload)
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_base64)
                    result = recognize_audio_bytes(audio_bytes)

                    if result.get("matched"):
                        response = {
                            "requestId": request_id,
                            "matched": True,
                            "trackId": result["song_id"],
                            "confidence": result["confidence"],
                            "margin": result["margin"],
                            "support": result["support"],
                            "topCandidates": result["top_candidates"],
                        }
                    else:
                        response = {
                            "requestId": request_id,
                            "matched": False,
                            "reason": result.get("reason", "low_confidence"),
                            "topCandidates": result.get("top_candidates", []),
                        }

                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to process ML fallback request %s", request_id)
                    response = {
                        "requestId": request_id,
                        "matched": False,
                        "error": str(exc),
                    }

                get_producer().send(ML_RESPONSE_TOPIC, response)

        except Exception:  # noqa: BLE001
            logger.exception("Kafka consumer connection failed, retrying in 5 seconds")
            time.sleep(5)


def _run_training_job(bucket: str, prefix: str) -> None:
    global train_in_progress

    with train_lock:
        if train_in_progress:
            logger.warning("Training already in progress; skipping")
            return
        train_in_progress = True

    update_job_status(
        current_job="train-from-s3",
        phase="training",
        started_at=utc_now_iso(),
        finished_at=None,
        last_error=None,
    )
    reset_job_progress()

    try:
        with log_stage(logger, "train-from-s3", bucket=bucket, prefix=prefix):
            train_from_s3_impl(bucket=bucket, prefix=prefix)

        update_job_status(
            phase="done",
            finished_at=utc_now_iso(),
            last_success=utc_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        update_job_status(
            phase="failed",
            finished_at=utc_now_iso(),
            last_error=str(exc),
        )
        logger.exception("Training failed")
    finally:
        train_in_progress = False


def _run_build_index_job(bucket: str, prefix: str) -> None:
    global build_in_progress

    with build_lock:
        if build_in_progress:
            logger.warning("Build index already in progress; skipping")
            return
        build_in_progress = True

    update_job_status(
        current_job="build-index-from-s3",
        phase="embedding_build",
        started_at=utc_now_iso(),
        finished_at=None,
        last_error=None,
    )
    reset_job_progress()

    try:
        with log_stage(logger, "build-embeddings-from-s3", bucket=bucket, prefix=prefix):
            build_embeddings_from_s3(bucket=bucket, prefix=prefix)

        update_job_status(phase="index_build")

        with log_stage(logger, "build-faiss-index"):
            build_faiss_index()

        update_job_status(
            phase="done",
            finished_at=utc_now_iso(),
            last_success=utc_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        update_job_status(
            phase="failed",
            finished_at=utc_now_iso(),
            last_error=str(exc),
        )
        logger.exception("Embedding/index build failed")
    finally:
        build_in_progress = False


def _run_bootstrap_job(bucket: str, prefix: str) -> None:
    global bootstrap_in_progress

    with bootstrap_lock:
        if bootstrap_in_progress:
            logger.warning("Bootstrap already in progress; skipping")
            return
        bootstrap_in_progress = True

    update_job_status(
        current_job="bootstrap-from-s3",
        phase="training",
        started_at=utc_now_iso(),
        finished_at=None,
        last_error=None,
    )
    reset_job_progress()

    total_started = time.time()

    try:
        with log_stage(logger, "bootstrap", bucket=bucket, prefix=prefix):
            with log_stage(logger, "bootstrap.training", bucket=bucket, prefix=prefix):
                update_job_status(phase="training")
                train_from_s3_impl(bucket=bucket, prefix=prefix)

            with log_stage(logger, "bootstrap.embedding_build", bucket=bucket, prefix=prefix):
                update_job_status(phase="embedding_build")
                build_embeddings_from_s3(bucket=bucket, prefix=prefix)

            with log_stage(logger, "bootstrap.index_build"):
                update_job_status(phase="index_build")
                build_faiss_index()

            with log_stage(logger, "bootstrap.reload_runtime"):
                update_job_status(phase="reload_runtime")
                load_runtime_artifacts()

        elapsed = time.time() - total_started
        logger.info("Bootstrap finished successfully total_elapsed_sec=%.2f", elapsed)

        update_job_status(
            phase="done",
            finished_at=utc_now_iso(),
            last_success=utc_now_iso(),
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - total_started
        logger.exception("Bootstrap failed total_elapsed_sec=%.2f", elapsed)

        update_job_status(
            phase="failed",
            finished_at=utc_now_iso(),
            last_error=str(exc),
        )
    finally:
        bootstrap_in_progress = False

@app.on_event("startup")
def startup() -> None:
    try:
        load_runtime_artifacts()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load runtime artifacts on startup; degraded mode enabled")

    if os.getenv("DISABLE_KAFKA_CONSUMER", "false").lower() != "true":
        threading.Thread(target=consume_requests_forever, daemon=True).start()


@app.on_event("shutdown")
def shutdown_resources() -> None:
    if producer is not None:
        producer.flush()
        producer.close()


atexit.register(shutdown_resources)


@app.post("/admin/train-from-s3", response_model=AsyncJobResponse)
async def train_from_s3_endpoint(payload: TrainFromS3Request, background_tasks: BackgroundTasks):
    if train_in_progress:
        raise HTTPException(status_code=409, detail="Training is already in progress")

    background_tasks.add_task(_run_training_job, payload.bucket, payload.prefix)
    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        bucket=payload.bucket,
        prefix=payload.prefix,
    )


@app.post("/admin/build-index-from-s3", response_model=AsyncJobResponse)
async def build_index_from_s3_endpoint(payload: TrainFromS3Request, background_tasks: BackgroundTasks):
    if build_in_progress:
        raise HTTPException(status_code=409, detail="Index build is already in progress")

    background_tasks.add_task(_run_build_index_job, payload.bucket, payload.prefix)
    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        bucket=payload.bucket,
        prefix=payload.prefix,
    )


@app.post("/admin/bootstrap-from-s3", response_model=AsyncJobResponse)
async def bootstrap_from_s3_endpoint(payload: TrainFromS3Request, background_tasks: BackgroundTasks):
    if bootstrap_in_progress:
        raise HTTPException(status_code=409, detail="Bootstrap is already in progress")

    background_tasks.add_task(_run_bootstrap_job, payload.bucket, payload.prefix)
    return AsyncJobResponse(
        accepted=True,
        status="scheduled",
        bucket=payload.bucket,
        prefix=payload.prefix,
    )


@app.post("/admin/reload-runtime")
async def reload_runtime_endpoint():
    try:
        load_runtime_artifacts()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ready = model is not None and index is not None and song_ids is not None
    return {
        "reloaded": True,
        "runtime_ready": ready,
    }


@app.get("/admin/status")
async def admin_status():
    with job_status_lock:
        current = dict(job_status)

    return {
        "train_in_progress": train_in_progress,
        "build_in_progress": build_in_progress,
        "bootstrap_in_progress": bootstrap_in_progress,
        "runtime_ready": model is not None and index is not None and song_ids is not None,
        "model_exists": os.path.exists(MODEL_PATH),
        "index_exists": os.path.exists(FAISS_INDEX_PATH),
        "song_ids_exists": os.path.exists(SONG_IDS_PATH),
        "job_status": current,
    }


@app.post("/recognition/resolve", response_model=RecognitionResponse)
async def recognition_resolve(file: UploadFile):
    audio_bytes = await file.read()

    try:
        result = recognize_audio_bytes(audio_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RecognitionResponse(**result)


@app.post("/recognize")
async def recognize(file: UploadFile):
    audio_bytes = await file.read()

    try:
        result = recognize_audio_bytes(audio_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@app.get("/health")
async def health():
    runtime_ready = model is not None and index is not None and song_ids is not None

    return {
        "status": "ok" if runtime_ready else "degraded",
        "runtime_ready": runtime_ready,
        "device": str(device),
        "model_exists": os.path.exists(MODEL_PATH),
        "index_exists": os.path.exists(FAISS_INDEX_PATH),
        "song_ids_exists": os.path.exists(SONG_IDS_PATH),
        "train_in_progress": train_in_progress,
        "build_in_progress": build_in_progress,
        "bootstrap_in_progress": bootstrap_in_progress,
        "index_total": index.ntotal if index is not None else None,
        "song_ids_total": len(song_ids) if song_ids is not None else None,
        "index_dim": index.d if index is not None else None,
        "model_dim": model.head[-1].out_features if model is not None else None,
    }