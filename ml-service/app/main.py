import logging

from fastapi import FastAPI, HTTPException, UploadFile

from app.logging_utils import configure_logging
from app.recognition import recognize_audio_bytes
from app.runtime import load_runtime_artifacts, runtime_ready, runtime_status
from app.schemas import RecognitionResponse, ReloadResponse
from pipeline.job_status import load_status_from_disk

app = FastAPI(title="ML Music Recognition inference")

configure_logging()
logger = logging.getLogger("ml-service")


@app.on_event("startup")
def startup() -> None:
    load_status_from_disk()

    try:
        load_runtime_artifacts()
    except Exception:
        logger.exception("Failed to load runtime artifacts on startup; degraded mode enabled")


@app.post("/admin/reload-runtime", response_model=ReloadResponse)
async def reload_runtime_endpoint():
    try:
        load_runtime_artifacts()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ReloadResponse(
        reloaded=True,
        runtime_ready=runtime_ready(),
    )


@app.get("/admin/status")
async def admin_status():
    return {
        **runtime_status(),
        "job_status": load_status_from_disk(),
    }


@app.post("/recognition/resolve", response_model=RecognitionResponse)
async def recognition_resolve(file: UploadFile):
    audio_bytes = await file.read()

    try:
        result = recognize_audio_bytes(audio_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RecognitionResponse(**result)


@app.post("/recognize")
async def recognize(file: UploadFile):
    audio_bytes = await file.read()

    try:
        return recognize_audio_bytes(audio_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health():
    status = runtime_status()

    return {
        "status": "ok" if status["runtime_ready"] else "degraded",
        **status,
    }
