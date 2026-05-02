import logging
import time

from app.clients import recognize_fingerprint, recognize_ml
from app.config import (
    COMBINED_RESULTS_PATH,
    EVAL_QUERIES_PATH,
    FINGERPRINT_RESULTS_PATH,
    FINGERPRINT_SERVICE_URL,
    ML_RESULTS_PATH,
    ML_SERVICE_URL,
    RUNNER_PROGRESS_EVERY,
)
from app.logging_utils import log_stage
from app.utils import read_jsonl, safe_int, write_jsonl

logger = logging.getLogger("evaluation.runners")


def _normalize_fingerprint_response(response: dict) -> dict:
    debug = response.get("debug") or {}
    top_candidates = debug.get("top_candidates") or []

    return {
        "matched": bool(response.get("match", response.get("matched", False))),
        "predicted_track_id": safe_int(response.get("track_id") or response.get("trackId")),
        "confidence": response.get("confidence"),
        "reason": response.get("reason"),
        "top_candidates": top_candidates,
        "raw_response": response,
    }


def _normalize_ml_response(response: dict) -> dict:
    top_candidates = response.get("top_candidates") or response.get("topCandidates") or []

    return {
        "matched": bool(response.get("matched", False)),
        "predicted_track_id": safe_int(
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


def _correct_top5(prediction: dict, gt_track_id: int | None, is_positive: bool) -> bool:
    if not is_positive or gt_track_id is None:
        return False

    ids = []
    for candidate in prediction.get("top_candidates", [])[:5]:
        ids.append(
            safe_int(
                candidate.get("track_id")
                or candidate.get("trackId")
                or candidate.get("song_id")
            )
        )

    return gt_track_id in ids


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


def _base_row(query: dict) -> dict:
    return {
        "query_id": query.get("query_id"),
        "query_path": query.get("query_path"),
        "bucket": query.get("bucket"),
        "corruption": query.get("corruption"),
        "duration_sec": query.get("duration_sec"),
        "is_positive": bool(query.get("is_positive")),
        "ground_truth_track_id": safe_int(query.get("track_id")),
        "audio_source": query.get("audio_source"),
        "s3_key": query.get("s3_key"),
    }


def _load_queries(limit: int | None) -> list[dict]:
    queries = read_jsonl(EVAL_QUERIES_PATH)

    if limit is not None:
        queries = queries[:limit]

    if not queries:
        raise ValueError(f"Eval queries file is empty: {EVAL_QUERIES_PATH}")

    return queries


def run_fingerprint_evaluation(
    limit: int | None = None,
    timeout_sec: float = 60.0,
) -> dict:
    with log_stage(
        logger,
        "run-fingerprint-evaluation",
        limit=limit,
        timeout_sec=timeout_sec,
        service_url=FINGERPRINT_SERVICE_URL,
    ):
        queries = _load_queries(limit)
        rows = []
        matched = 0
        correct = 0
        failed = 0
        started = time.perf_counter()

        for idx, query in enumerate(queries, start=1):
            base = _base_row(query)

            try:
                raw, latency_ms = recognize_fingerprint(
                    FINGERPRINT_SERVICE_URL,
                    query["query_path"],
                    timeout_sec,
                )
                pred = _normalize_fingerprint_response(raw)
            except Exception as exc:
                failed += 1
                latency_ms = None
                pred = _failed_prediction(exc)
                logger.warning(
                    "Fingerprint query failed idx=%s query_id=%s path=%s error=%s",
                    idx,
                    query.get("query_id"),
                    query.get("query_path"),
                    exc,
                )

            correct_top1 = _correct_top1(
                pred,
                base["ground_truth_track_id"],
                base["is_positive"],
            )
            correct_top5 = _correct_top5(
                pred,
                base["ground_truth_track_id"],
                base["is_positive"],
            )

            if pred["matched"]:
                matched += 1
            if correct_top1:
                correct += 1

            rows.append(
                {
                    **base,
                    "predicted_track_id": pred["predicted_track_id"],
                    "matched": pred["matched"],
                    "correct_top1": correct_top1,
                    "correct_top5": correct_top5,
                    "confidence": pred.get("confidence"),
                    "reason": pred.get("reason"),
                    "latency_ms": latency_ms,
                    "top_candidates": pred.get("top_candidates", []),
                    "raw_prediction": pred.get("raw_response"),
                }
            )

            if idx % RUNNER_PROGRESS_EVERY == 0 or idx == len(queries):
                elapsed = time.perf_counter() - started
                logger.info(
                    "Fingerprint eval progress queries=%s/%s matched=%s correct=%s failed=%s elapsed_sec=%.2f",
                    idx,
                    len(queries),
                    matched,
                    correct,
                    failed,
                    elapsed,
                )

        write_jsonl(FINGERPRINT_RESULTS_PATH, rows)

        summary = {
            "queries": len(rows),
            "matched": matched,
            "correct_top1": correct,
            "failed": failed,
            "results_path": str(FINGERPRINT_RESULTS_PATH),
        }

        logger.info("Fingerprint evaluation finished summary=%s", summary)
        return summary


def run_ml_evaluation(
    limit: int | None = None,
    timeout_sec: float = 60.0,
) -> dict:
    with log_stage(
        logger,
        "run-ml-evaluation",
        limit=limit,
        timeout_sec=timeout_sec,
        service_url=ML_SERVICE_URL,
    ):
        queries = _load_queries(limit)
        rows = []
        matched = 0
        correct = 0
        failed = 0
        started = time.perf_counter()

        for idx, query in enumerate(queries, start=1):
            base = _base_row(query)

            try:
                raw, latency_ms = recognize_ml(
                    ML_SERVICE_URL,
                    query["query_path"],
                    timeout_sec,
                )
                pred = _normalize_ml_response(raw)
            except Exception as exc:
                failed += 1
                latency_ms = None
                pred = _failed_prediction(exc)
                logger.warning(
                    "ML query failed idx=%s query_id=%s path=%s error=%s",
                    idx,
                    query.get("query_id"),
                    query.get("query_path"),
                    exc,
                )

            correct_top1 = _correct_top1(
                pred,
                base["ground_truth_track_id"],
                base["is_positive"],
            )
            correct_top5 = _correct_top5(
                pred,
                base["ground_truth_track_id"],
                base["is_positive"],
            )

            if pred["matched"]:
                matched += 1
            if correct_top1:
                correct += 1

            rows.append(
                {
                    **base,
                    "predicted_track_id": pred["predicted_track_id"],
                    "matched": pred["matched"],
                    "correct_top1": correct_top1,
                    "correct_top5": correct_top5,
                    "confidence": pred.get("confidence"),
                    "margin": pred.get("margin"),
                    "support": pred.get("support"),
                    "reason": pred.get("reason"),
                    "latency_ms": latency_ms,
                    "top_candidates": pred.get("top_candidates", []),
                    "raw_prediction": pred.get("raw_response"),
                }
            )

            if idx % RUNNER_PROGRESS_EVERY == 0 or idx == len(queries):
                elapsed = time.perf_counter() - started
                logger.info(
                    "ML eval progress queries=%s/%s matched=%s correct=%s failed=%s elapsed_sec=%.2f",
                    idx,
                    len(queries),
                    matched,
                    correct,
                    failed,
                    elapsed,
                )

        write_jsonl(ML_RESULTS_PATH, rows)

        summary = {
            "queries": len(rows),
            "matched": matched,
            "correct_top1": correct,
            "failed": failed,
            "results_path": str(ML_RESULTS_PATH),
        }

        logger.info("ML evaluation finished summary=%s", summary)
        return summary


def run_combined_evaluation(
    limit: int | None = None,
    timeout_sec: float = 60.0,
) -> dict:
    with log_stage(
        logger,
        "run-combined-evaluation",
        limit=limit,
        timeout_sec=timeout_sec,
        fingerprint_url=FINGERPRINT_SERVICE_URL,
        ml_url=ML_SERVICE_URL,
    ):
        queries = _load_queries(limit)
        rows = []

        fp_matched = 0
        fp_correct = 0
        ml_matched = 0
        ml_correct = 0
        failed = 0

        started = time.perf_counter()

        for idx, query in enumerate(queries, start=1):
            base = _base_row(query)

            try:
                fp_raw, fp_latency = recognize_fingerprint(
                    FINGERPRINT_SERVICE_URL,
                    query["query_path"],
                    timeout_sec,
                )
                fp = _normalize_fingerprint_response(fp_raw)
            except Exception as exc:
                failed += 1
                fp_latency = None
                fp = _failed_prediction(exc)
                logger.warning(
                    "Combined fingerprint query failed idx=%s query_id=%s error=%s",
                    idx,
                    query.get("query_id"),
                    exc,
                )

            try:
                ml_raw, ml_latency = recognize_ml(
                    ML_SERVICE_URL,
                    query["query_path"],
                    timeout_sec,
                )
                ml = _normalize_ml_response(ml_raw)
            except Exception as exc:
                failed += 1
                ml_latency = None
                ml = _failed_prediction(exc)
                logger.warning(
                    "Combined ML query failed idx=%s query_id=%s error=%s",
                    idx,
                    query.get("query_id"),
                    exc,
                )

            fp_correct_top1 = _correct_top1(
                fp,
                base["ground_truth_track_id"],
                base["is_positive"],
            )
            fp_correct_top5 = _correct_top5(
                fp,
                base["ground_truth_track_id"],
                base["is_positive"],
            )

            ml_correct_top1 = _correct_top1(
                ml,
                base["ground_truth_track_id"],
                base["is_positive"],
            )
            ml_correct_top5 = _correct_top5(
                ml,
                base["ground_truth_track_id"],
                base["is_positive"],
            )

            if fp["matched"]:
                fp_matched += 1
            if fp_correct_top1:
                fp_correct += 1

            if ml["matched"]:
                ml_matched += 1
            if ml_correct_top1:
                ml_correct += 1

            fp_pred = fp["predicted_track_id"]
            ml_pred = ml["predicted_track_id"]

            rows.append(
                {
                    **base,
                    "fingerprint": {
                        **fp,
                        "latency_ms": fp_latency,
                        "correct_top1": fp_correct_top1,
                        "correct_top5": fp_correct_top5,
                    },
                    "ml": {
                        **ml,
                        "latency_ms": ml_latency,
                        "correct_top1": ml_correct_top1,
                        "correct_top5": ml_correct_top5,
                    },
                    "agreement": {
                        "same_prediction": fp_pred is not None and fp_pred == ml_pred,
                        "both_correct": fp_correct_top1 and ml_correct_top1,
                        "fingerprint_only_correct": fp_correct_top1 and not ml_correct_top1,
                        "ml_only_correct": ml_correct_top1 and not fp_correct_top1,
                        "both_wrong": base["is_positive"] and not fp_correct_top1 and not ml_correct_top1,
                        "both_abstained": not fp["matched"] and not ml["matched"],
                    },
                }
            )

            if idx % RUNNER_PROGRESS_EVERY == 0 or idx == len(queries):
                elapsed = time.perf_counter() - started
                logger.info(
                    "Combined eval progress queries=%s/%s fp_matched=%s fp_correct=%s ml_matched=%s ml_correct=%s failed=%s elapsed_sec=%.2f",
                    idx,
                    len(queries),
                    fp_matched,
                    fp_correct,
                    ml_matched,
                    ml_correct,
                    failed,
                    elapsed,
                )

        write_jsonl(COMBINED_RESULTS_PATH, rows)

        summary = {
            "queries": len(rows),
            "fingerprint_matched": fp_matched,
            "fingerprint_correct_top1": fp_correct,
            "ml_matched": ml_matched,
            "ml_correct_top1": ml_correct,
            "failed": failed,
            "results_path": str(COMBINED_RESULTS_PATH),
        }

        logger.info("Combined evaluation finished summary=%s", summary)
        return summary