import time
from pathlib import Path

import requests


def recognize_with_fingerprint(
    fingerprint_url: str,
    audio_path: str,
    timeout_sec: float = 60.0,
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
            f"{fingerprint_url.rstrip('/')}/recognize",
            files=files,
            timeout=timeout_sec,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()

    return response.json(), latency_ms