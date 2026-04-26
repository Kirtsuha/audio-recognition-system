import atexit
import base64
import json
import logging
import os
import threading
import time

from fastapi import FastAPI, HTTPException, UploadFile
from kafka import KafkaConsumer, KafkaProducer

from app.logging_utils import configure_logging
from app.recognition import recognize_audio_bytes
from app.runtime import load_runtime_artifacts, runtime_ready, runtime_status
from app.schemas import RecognitionResponse, ReloadResponse
from pipeline.job_status import load_status_from_disk

app = FastAPI(title="ML Music Recognition inference")

configure_logging()
logger = logging.getLogger("ml-service")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
ML_REQUEST_TOPIC = os.getenv("ML_REQUEST_TOPIC", "ml-recognition-requests")
ML_RESPONSE_TOPIC = os.getenv("ML_RESPONSE_TOPIC", "ml-recognition-responses")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "ml-service")

producer: KafkaProducer | None = None


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

                except Exception as exc:
                    logger.exception("Failed to process ML fallback request %s", request_id)
                    response = {
                        "requestId": request_id,
                        "matched": False,
                        "error": str(exc),
                    }

                get_producer().send(ML_RESPONSE_TOPIC, response)

        except Exception:
            logger.exception("Kafka consumer connection failed, retrying in 5 seconds")
            time.sleep(5)


@app.on_event("startup")
def startup() -> None:
    load_status_from_disk()

    try:
        load_runtime_artifacts()
    except Exception:
        logger.exception("Failed to load runtime artifacts on startup; degraded mode enabled")

    if os.getenv("DISABLE_KAFKA_CONSUMER", "false").lower() != "true":
        threading.Thread(target=consume_requests_forever, daemon=True).start()


@app.on_event("shutdown")
def shutdown_resources() -> None:
    if producer is not None:
        producer.flush()
        producer.close()


atexit.register(shutdown_resources)


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