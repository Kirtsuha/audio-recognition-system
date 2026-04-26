import logging
import os
import tempfile
import time
from typing import Any

from app.inference import aggregate_results
from app.runtime import embed_windows, get_runtime, should_accept
from pipeline.config import FAISS_TOP_K
from pipeline.dataset import extract_sliding_windows, load_audio

logger = logging.getLogger("ml-service.recognition")


def recognize_audio_bytes(audio_bytes: bytes) -> dict[str, Any]:
    runtime = get_runtime()

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
        scores, nearest = runtime.index.search(embs, k=FAISS_TOP_K)

        result = aggregate_results(
            scores=scores,
            indices=nearest,
            song_ids=runtime.song_ids,
        )

        if result is None:
            logger.info(
                "Recognition finished matched=false reason=no_candidates windows=%s elapsed_sec=%.2f",
                len(windows),
                time.time() - started,
            )
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