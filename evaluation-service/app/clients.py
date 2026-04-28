import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger("evaluation.clients")


def post_audio_file(
    url: str,
    audio_path: str,
    timeout_sec: float,
) -> tuple[dict, float]:
    started = time.perf_counter()

    with open(audio_path, "rb") as f:
        files = {
            "file": (
                Path(audio_path).name,
                f,
                "audio/wav",
            )
        }

        response = requests.post(
            url,
            files=files,
            timeout=timeout_sec,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0

    if response.status_code >= 400:
        logger.warning(
            "Audio request failed url=%s path=%s status=%s latency_ms=%.2f body=%s",
            url,
            audio_path,
            response.status_code,
            latency_ms,
            response.text[:500],
        )

    response.raise_for_status()

    return response.json(), latency_ms


def recognize_fingerprint(
    service_url: str,
    audio_path: str,
    timeout_sec: float,
) -> tuple[dict, float]:
    return post_audio_file(
        url=f"{service_url.rstrip('/')}/recognize",
        audio_path=audio_path,
        timeout_sec=timeout_sec,
    )


def recognize_ml(
    service_url: str,
    audio_path: str,
    timeout_sec: float,
) -> tuple[dict, float]:
    return post_audio_file(
        url=f"{service_url.rstrip('/')}/recognize",
        audio_path=audio_path,
        timeout_sec=timeout_sec,
    )