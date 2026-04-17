import atexit
import base64
import json
import logging
import os
import tempfile
import threading
import time

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from kafka import KafkaConsumer, KafkaProducer

from app.inference import aggregate_results
from app.model import AudioEncoder
from pipeline.config import MODEL_PATH
from pipeline.dataset import load_audio, random_segment
from pipeline.to_mel import to_mel

app = FastAPI()
logger = logging.getLogger("ml-service")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
ML_REQUEST_TOPIC = os.getenv("ML_REQUEST_TOPIC", "ml-recognition-requests")
ML_RESPONSE_TOPIC = os.getenv("ML_RESPONSE_TOPIC", "ml-recognition-responses")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "ml-service")

model = AudioEncoder()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

index = faiss.read_index("data/faiss.index")
song_ids = np.load("data/song_ids.npy")
producer = None


def recognize_audio_bytes(audio_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    try:
        audio = load_audio(temp_path)
        all_indices = []

        for _ in range(5):
            seg = random_segment(audio)
            mel = to_mel(seg).unsqueeze(0)
            emb = model(mel).detach().numpy().astype("float32")
            _, nearest = index.search(emb, k=5)
            all_indices.append(nearest[0])

        result = aggregate_results(all_indices, song_ids)
        if result is None:
            raise ValueError("ML model could not produce a match")
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
        )
    return producer


def consume_requests_forever() -> None:
    while True:
        try:
            consumer = KafkaConsumer(
                ML_REQUEST_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=KAFKA_CONSUMER_GROUP,
                auto_offset_reset="earliest",
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
                    result = recognize_audio_bytes(base64.b64decode(audio_base64))
                    response = {
                        "requestId": request_id,
                        "trackId": result["song_id"],
                        "confidence": result["confidence"],
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to process ML fallback request %s", request_id)
                    response = {
                        "requestId": request_id,
                        "error": str(exc),
                    }

                get_producer().send(ML_RESPONSE_TOPIC, response)
                get_producer().flush()
        except Exception:  # noqa: BLE001
            logger.exception("Kafka consumer connection failed, retrying in 5 seconds")
            time.sleep(5)


@app.on_event("startup")
def start_kafka_consumer() -> None:
    if os.getenv("DISABLE_KAFKA_CONSUMER", "false").lower() == "true":
        logger.info("Kafka consumer disabled by configuration")
        return

    threading.Thread(target=consume_requests_forever, daemon=True).start()


@app.on_event("shutdown")
def shutdown_resources() -> None:
    if producer is not None:
        producer.flush()
        producer.close()


atexit.register(shutdown_resources)


@app.post("/recognize")
async def recognize(file: UploadFile):
    audio_bytes = await file.read()

    try:
        result = recognize_audio_bytes(audio_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "track_id": result["song_id"],
        "confidence": result["confidence"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
