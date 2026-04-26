import logging
import os
import tempfile
import time

import faiss
import numpy as np
import torch

from app.inference import aggregate_results
from app.model import AudioEncoder
from pipeline.config import (
    MODEL_PATH,
    FAISS_INDEX_PATH,
    SONG_IDS_PATH,
    FAISS_TOP_K,
    EVAL_QUERIES_PATH,
    EVAL_RESULTS_PATH,
)
from pipeline.dataset import extract_sliding_windows, load_audio
from evaluation.eval_utils import read_jsonl, write_jsonl
from pipeline.to_mel import to_mel

logger = logging.getLogger("ml-pipeline.eval-runner")


def _load_runtime():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioEncoder().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    song_ids = np.load(SONG_IDS_PATH)

    if index.ntotal != len(song_ids):
        raise RuntimeError(
            f"Inconsistent runtime artifacts: index.ntotal={index.ntotal}, song_ids={len(song_ids)}"
        )

    return device, model, index, song_ids


def _embed_windows(model: AudioEncoder, device: torch.device, windows: list[np.ndarray]) -> np.ndarray:
    batch = torch.stack([to_mel(w) for w in windows]).to(device)

    with torch.inference_mode():
        embs = model(batch).detach().cpu().numpy().astype("float32")

    faiss.normalize_L2(embs)
    return embs


def _recognize_query_file(
    model: AudioEncoder,
    device: torch.device,
    index: faiss.Index,
    song_ids: np.ndarray,
    query_path: str,
) -> dict:
    audio = load_audio(query_path)
    windows = extract_sliding_windows(audio)
    if not windows:
        return {
            "matched": False,
            "reason": "no_windows",
            "top_candidates": [],
        }

    embs = _embed_windows(model, device, windows)
    scores, nearest = index.search(embs, k=FAISS_TOP_K)

    result = aggregate_results(scores=scores, indices=nearest, song_ids=song_ids)
    if result is None:
        return {
            "matched": False,
            "reason": "no_candidates",
            "top_candidates": [],
        }

    result["matched"] = True
    return result


def run_evaluation(warmup_queries: int = 3, limit: int | None = None) -> dict:
    queries = read_jsonl(EVAL_QUERIES_PATH)
    if not queries:
        raise ValueError("Eval queries file is empty")

    if limit is not None:
        queries = queries[:limit]

    device, model, index, song_ids = _load_runtime()

    # Warmup
    for row in queries[:warmup_queries]:
        try:
            _recognize_query_file(model, device, index, song_ids, row["query_path"])
        except Exception:
            logger.exception("Warmup query failed query_id=%s", row["query_id"])

    results = []

    for idx, row in enumerate(queries, start=1):
        started = time.perf_counter()

        try:
            prediction = _recognize_query_file(model, device, index, song_ids, row["query_path"])
            latency_ms = (time.perf_counter() - started) * 1000.0

            top_candidates = prediction.get("top_candidates", [])
            predicted_track_id = prediction.get("song_id")
            top5_ids = [int(x["song_id"]) for x in top_candidates[:5]]

            gt_track_id = row["track_id"]
            is_positive = bool(row["is_positive"])

            correct_top1 = bool(is_positive and predicted_track_id == gt_track_id)
            correct_top5 = bool(is_positive and gt_track_id in top5_ids)

            result_row = {
                "query_id": row["query_id"],
                "query_path": row["query_path"],
                "bucket": row["bucket"],
                "corruption": row["corruption"],
                "duration_sec": row["duration_sec"],
                "is_positive": is_positive,
                "ground_truth_track_id": gt_track_id,
                "predicted_track_id": predicted_track_id,
                "matched": bool(prediction.get("matched", False)),
                "correct_top1": correct_top1,
                "correct_top5": correct_top5,
                "confidence": prediction.get("confidence"),
                "margin": prediction.get("margin"),
                "support": prediction.get("support"),
                "reason": prediction.get("reason"),
                "latency_ms": latency_ms,
                "top_candidates": top_candidates,
            }
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000.0
            result_row = {
                "query_id": row["query_id"],
                "query_path": row["query_path"],
                "bucket": row["bucket"],
                "corruption": row["corruption"],
                "duration_sec": row["duration_sec"],
                "is_positive": bool(row["is_positive"]),
                "ground_truth_track_id": row["track_id"],
                "predicted_track_id": None,
                "matched": False,
                "correct_top1": False,
                "correct_top5": False,
                "confidence": None,
                "margin": None,
                "support": None,
                "reason": f"exception:{exc}",
                "latency_ms": latency_ms,
                "top_candidates": [],
            }

        results.append(result_row)

        if idx % 100 == 0 or idx == len(queries):
            logger.info("Eval progress queries=%s/%s", idx, len(queries))

    write_jsonl(EVAL_RESULTS_PATH, results)

    summary = {
        "queries": len(results),
        "results_path": str(EVAL_RESULTS_PATH),
    }
    logger.info("Eval run finished summary=%s", summary)
    return summary