from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.reranker import RerankerEngine
from app.s3_storage import S3Storage
from app.schemas import (
    HealthResponse,
    ReadyResponse,
    RerankRequest,
    RerankResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ml-reranker.main")

app = FastAPI(title="ML Reranker Service", version="1.0.0")

storage = S3Storage()
engine = RerankerEngine(storage=storage)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = engine.status()
    return HealthResponse(
        status="ok" if status.model_loaded else "degraded",
        service=settings.service_name,
        model_loaded=status.model_loaded,
        device=status.device,
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    status = engine.status()
    return ReadyResponse(
        ready=status.model_loaded,
        model_loaded=status.model_loaded,
        model_path=status.model_path,
        query_audio_bucket=settings.query_audio_bucket,
        reference_audio_bucket=settings.reference_audio_bucket,
        sr=settings.sr,
        segment_seconds=settings.segment_seconds,
        max_candidates=settings.max_candidates,
        reference_store_loaded=status.reference_store_loaded,
        reference_store_vectors=status.reference_store_vectors,
        reference_store_tracks=status.reference_store_tracks,
    )


@app.post("/admin/reload-model")
async def reload_model() -> dict:
    try:
        engine.load_model()
        status = engine.status()
        return {
            "reloaded": True,
            "model_loaded": status.model_loaded,
            "model_path": status.model_path,
            "device": status.device,
        }
    except Exception as exc:
        logger.exception("Failed to reload model")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/admin/reload-reference-store")
async def reload_reference_store() -> dict:
    try:
        engine.load_reference_store()
        status = engine.status()
        return {
            "reloaded": True,
            "reference_store_loaded": status.reference_store_loaded,
            "reference_store_vectors": status.reference_store_vectors,
            "reference_store_tracks": status.reference_store_tracks,
        }
    except Exception as exc:
        logger.exception("Failed to reload reference store")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    try:
        return engine.rerank(
            request_id=req.request_id,
            query_bucket=req.query_audio.bucket,
            query_key=req.query_audio.key,
            reference_bucket=req.reference_bucket,
            fingerprint=req.fingerprint,
            options=req.options,
        )
    except RuntimeError as exc:
        logger.exception("Rerank runtime error request_id=%s", req.request_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Rerank failed request_id=%s", req.request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
