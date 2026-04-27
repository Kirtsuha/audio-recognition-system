import logging
import os
import time
from pathlib import Path

from evaluation.eval_runner import _load_runtime, _recognize_query_file
from evaluation.eval_utils import read_jsonl, write_jsonl
from evaluation.fingerprint_client import recognize_with_fingerprint
from pipeline.config import EVAL_QUERIES_PATH

logger = logging.getLogger("ml-pipeline.combined-eval")

FINGERPRINT_SERVICE_URL = os.getenv(
    "FINGERPRINT_SERVICE_URL",
    "http://fingerprint-service:8000",
)

COMBINED_RESULTS_PATH = os.getenv(
    "COMBINED_RESULTS_PATH",
    "/app/artifacts/eval/combined_eval_results.jsonl",
)


def _norm_track_id(value):
    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def _normalize_fp_response(response: dict) -> dict:
    return {
        "matched": bool(response.get("match", response.get("matched", False))),
        "predicted_track_id": _norm_track_id(
            response.get("track_id") or response.get("trackId")
        ),
        "confidence": response.get("confidence"),
        "reason": response.get("reason"),
        "raw_response": response,
    }


def _normalize_ml_response(response: dict) -> dict:
    top_candidates = response.get("top_candidates") or response.get("topCandidates") or []

    return {
        "matched": bool(response.get("matched", False)),
        "predicted_track_id": _norm_track_id(
            response.get("song_id")
            or response.get("track_id")
            or response.get("trackId")
        ),
        "confidence": response.get("confidence"),
        "margin": response.get("margin"),
        "support": response.get("support"),
        "reason": response.get("reason"),
        "top_candidates": top_candidates,
        "raw_response": response,
    }


def _correct_top1(prediction: dict, gt_track_id: int | None, is_positive: bool) -> bool:
    return bool(
        is_positive
        and prediction.get("matched")
        and prediction.get("predicted_track_id") is not None
        and gt_track_id is not None
        and int(prediction["predicted_track_id"]) == int(gt_track_id)
    )


def _correct_top5_ml(prediction: dict, gt_track_id: int | None, is_positive: bool) -> bool:
    if not is_positive or gt_track_id is None:
        return False

    top_ids = []

    for candidate in prediction.get("top_candidates", [])[:5]:
        top_ids.append(
            _norm_track_id(
                candidate.get("song_id")
                or candidate.get("track_id")
                or candidate.get("trackId")
            )
        )

    return gt_track_id in top_ids


def _failed_prediction(exc: Exception) -> dict:
    return {
        "matched": False,
        "predicted_track_id": None,
        "confidence": None,
        "margin": None,
        "support": None,
        "reason": f"exception:{exc}",
        "top_candidates": [],
        "raw_response": None,
    }


def run_combined_evaluation(
    limit: int | None = None,
    fingerprint_url: str | None = None,
    fingerprint_timeout_sec: float = 60.0,
) -> dict:
    queries = read_jsonl(EVAL_QUERIES_PATH)

    if not queries:
        raise ValueError(f"Eval queries file is empty: {EVAL_QUERIES_PATH}")

    if limit is not None:
        queries = queries[:limit]

    fingerprint_url = fingerprint_url or FINGERPRINT_SERVICE_URL

    device, model, index, song_ids = _load_runtime()

    rows = []

    for idx, query in enumerate(queries, start=1):
        query_path = query["query_path"]

        if not Path(query_path).exists():
            logger.warning("Query audio file not found: %s", query_path)

        gt_track_id = _norm_track_id(query.get("track_id"))
        is_positive = bool(query.get("is_positive"))

        ml_latency_ms = None
        fp_latency_ms = None

        try:
            started = time.perf_counter()

            ml_raw = _recognize_query_file(
                model=model,
                device=device,
                index=index,
                song_ids=song_ids,
                query_path=query_path,
            )

            ml_latency_ms = (time.perf_counter() - started) * 1000.0
            ml_prediction = _normalize_ml_response(ml_raw)

        except Exception as exc:
            logger.exception("ML eval failed query_id=%s", query.get("query_id"))
            ml_prediction = _failed_prediction(exc)

        try:
            fp_raw, fp_latency_ms = recognize_with_fingerprint(
                fingerprint_url=fingerprint_url,
                audio_path=query_path,
                timeout_sec=fingerprint_timeout_sec,
            )

            fp_prediction = _normalize_fp_response(fp_raw)

        except Exception as exc:
            logger.exception("Fingerprint eval failed query_id=%s", query.get("query_id"))
            fp_prediction = _failed_prediction(exc)

        ml_correct_top1 = _correct_top1(ml_prediction, gt_track_id, is_positive)
        ml_correct_top5 = _correct_top5_ml(ml_prediction, gt_track_id, is_positive)

        fp_correct_top1 = _correct_top1(fp_prediction, gt_track_id, is_positive)

        ml_predicted_track_id = ml_prediction.get("predicted_track_id")
        fp_predicted_track_id = fp_prediction.get("predicted_track_id")

        row = {
            "query_id": query["query_id"],
            "query_path": query_path,
            "bucket": query["bucket"],
            "corruption": query["corruption"],
            "duration_sec": query["duration_sec"],
            "is_positive": is_positive,
            "ground_truth_track_id": gt_track_id,

            "ml": {
                **ml_prediction,
                "latency_ms": ml_latency_ms,
                "correct_top1": ml_correct_top1,
                "correct_top5": ml_correct_top5,
            },

            "fingerprint": {
                **fp_prediction,
                "latency_ms": fp_latency_ms,
                "correct_top1": fp_correct_top1,
                "correct_top5": fp_correct_top1,
            },

            "agreement": {
                "same_prediction": (
                    ml_predicted_track_id is not None
                    and fp_predicted_track_id is not None
                    and ml_predicted_track_id == fp_predicted_track_id
                ),
                "both_correct": ml_correct_top1 and fp_correct_top1,
                "ml_only_correct": ml_correct_top1 and not fp_correct_top1,
                "fingerprint_only_correct": fp_correct_top1 and not ml_correct_top1,
                "both_wrong": is_positive and not ml_correct_top1 and not fp_correct_top1,
                "both_abstained": (
                    not ml_prediction.get("matched")
                    and not fp_prediction.get("matched")
                ),
            },
        }

        rows.append(row)

        if idx % 100 == 0 or idx == len(queries):
            logger.info(
                "Combined eval progress queries=%s/%s",
                idx,
                len(queries),
            )

    write_jsonl(COMBINED_RESULTS_PATH, rows)

    return {
        "queries": len(rows),
        "results_path": COMBINED_RESULTS_PATH,
        "fingerprint_url": fingerprint_url,
    }