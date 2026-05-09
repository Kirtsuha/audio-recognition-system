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


def retrieve_fingerprint_candidates(
    service_url: str,
    audio_path: str,
    timeout_sec: float,
    top_k: int,
) -> tuple[dict, float]:
    return post_audio_file(
        url=f"{service_url.rstrip('/')}/retrieve-candidates?top_k={top_k}",
        audio_path=audio_path,
        timeout_sec=timeout_sec,
    )


def _auth_payload(username: str, password: str) -> dict:
    return {"username": username, "password": password}


def get_orchestration_token(
    service_url: str,
    username: str,
    password: str,
    timeout_sec: float,
) -> str:
    base = service_url.rstrip("/")
    payload = _auth_payload(username, password)

    login = requests.post(
        f"{base}/api/auth/login",
        json=payload,
        timeout=timeout_sec,
    )

    if login.status_code == 401:
        register = requests.post(
            f"{base}/api/auth/register",
            json=payload,
            timeout=timeout_sec,
        )
        if register.status_code >= 400 and "already" not in register.text.lower():
            register.raise_for_status()

        login = requests.post(
            f"{base}/api/auth/login",
            json=payload,
            timeout=timeout_sec,
        )

    login.raise_for_status()
    body = login.json()
    token = body.get("accessToken") or body.get("access_token") or body.get("token")

    if not token:
        raise RuntimeError(f"Orchestration auth response has no token: {body}")

    return str(token)


def recognize_orchestration(
    service_url: str,
    audio_path: str,
    timeout_sec: float,
    token: str,
) -> tuple[dict, float]:
    started = time.perf_counter()
    url = f"{service_url.rstrip('/')}/api/recognition"

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
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_sec,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0

    if response.status_code >= 400:
        logger.warning(
            "Orchestration recognition failed url=%s path=%s status=%s latency_ms=%.2f body=%s",
            url,
            audio_path,
            response.status_code,
            latency_ms,
            response.text[:500],
        )

    response.raise_for_status()
    return response.json(), latency_ms
