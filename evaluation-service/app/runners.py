import logging
import time

from app.clients import (
    get_orchestration_token,
    recognize_orchestration,
    retrieve_fingerprint_candidates,
)
from app.config import (
    COMBINED_RESULTS_PATH,
    EVALUATION_PASSWORD,
    EVALUATION_USERNAME,
    EVAL_QUERIES_PATH,
    FINGERPRINT_RESULTS_PATH,
    FINGERPRINT_SERVICE_URL,
    ORCHESTRATION_SERVICE_URL,
    RECOGNITION_RESULTS_PATH,
    RUNNER_PROGRESS_EVERY,
)
from app.logging_utils import log_stage
from app.utils import read_jsonl, safe_int, write_jsonl

logger = logging.getLogger("evaluation.runners")


def _normalize_fingerprint_response(response: dict) -> dict:
    debug = response.get("debug") or {}
    top_candidates = (
        response.get("candidates")
        or response.get("top_candidates")
        or debug.get("top_candidates")
        or []
    )
    best = top_candidates[0] if top_candidates else {}

    return {
        "matched": bool(response.get("match", response.get("matched", False))),
        "predicted_track_id": safe_int(
            response.get("track_id")
            or response.get("trackId")
            or best.get("track_id")
            or best.get("trackId")
        ),
        "confidence": response.get("confidence") if response.get("confidence") is not None else best.get("confidence"),
        "best_aligned_matches": response.get("best_aligned_matches") or response.get("bestAlignedMatches"),
        "second_aligned_matches": response.get("second_aligned_matches") or response.get("secondAlignedMatches"),
        "reason": response.get("reason"),
        "top_candidates": top_candidates,
        "raw_response": response,
    }


def _normalize_recognition_response(response: dict) -> dict:
    return {
        "matched": bool(response.get("match", response.get("matched", False))),
        "predicted_track_id": safe_int(
            response.get("trackId")
            or response.get("track_id")
            or response.get("song_id")
        ),
        "confidence": response.get("confidence"),
        "source": response.get("source"),
        "title": response.get("title"),
        "artist": response.get("artist"),
        "reason": response.get("reason"),
        "top_candidates": response.get("top_candidates") or response.get("candidates") or [],
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
    top_k: int = 50,
) -> dict:
    with log_stage(
        logger,
        "run-fingerprint-evaluation",
        limit=limit,
        timeout_sec=timeout_sec,
        service_url=FINGERPRINT_SERVICE_URL,
        top_k=top_k,
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
                raw, latency_ms = retrieve_fingerprint_candidates(
                    FINGERPRINT_SERVICE_URL,
                    query["query_path"],
                    timeout_sec,
                    top_k,
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
                    "best_aligned_matches": pred.get("best_aligned_matches"),
                    "second_aligned_matches": pred.get("second_aligned_matches"),
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
            "top_k": top_k,
        }

        logger.info("Fingerprint evaluation finished summary=%s", summary)
        return summary


def run_recognition_evaluation(
    limit: int | None = None,
    timeout_sec: float = 60.0,
) -> dict:
    with log_stage(
        logger,
        "run-recognition-evaluation",
        limit=limit,
        timeout_sec=timeout_sec,
        service_url=ORCHESTRATION_SERVICE_URL,
    ):
        token = get_orchestration_token(
            ORCHESTRATION_SERVICE_URL,
            EVALUATION_USERNAME,
            EVALUATION_PASSWORD,
            timeout_sec,
        )
        queries = _load_queries(limit)
        rows = []
        matched = 0
        correct = 0
        failed = 0
        source_counts = {}
        started = time.perf_counter()

        for idx, query in enumerate(queries, start=1):
            base = _base_row(query)

            try:
                raw, latency_ms = recognize_orchestration(
                    ORCHESTRATION_SERVICE_URL,
                    query["query_path"],
                    timeout_sec,
                    token,
                )
                pred = _normalize_recognition_response(raw)
            except Exception as exc:
                failed += 1
                latency_ms = None
                pred = _failed_prediction(exc)
                pred["source"] = "exception"
                logger.warning(
                    "Recognition query failed idx=%s query_id=%s path=%s error=%s",
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
            source = pred.get("source") or "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1

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
                    "confidence": pred.get("confidence"),
                    "source": source,
                    "title": pred.get("title"),
                    "artist": pred.get("artist"),
                    "reason": pred.get("reason"),
                    "latency_ms": latency_ms,
                    "top_candidates": pred.get("top_candidates", []),
                    "raw_prediction": pred.get("raw_response"),
                }
            )

            if idx % RUNNER_PROGRESS_EVERY == 0 or idx == len(queries):
                elapsed = time.perf_counter() - started
                logger.info(
                    "Recognition eval progress queries=%s/%s matched=%s correct=%s failed=%s sources=%s elapsed_sec=%.2f",
                    idx,
                    len(queries),
                    matched,
                    correct,
                    failed,
                    source_counts,
                    elapsed,
                )

        write_jsonl(RECOGNITION_RESULTS_PATH, rows)

        summary = {
            "queries": len(rows),
            "matched": matched,
            "correct_top1": correct,
            "failed": failed,
            "source_counts": source_counts,
            "results_path": str(RECOGNITION_RESULTS_PATH),
        }

        logger.info("Recognition evaluation finished summary=%s", summary)
        return summary


def run_combined_evaluation(
    limit: int | None = None,
    timeout_sec: float = 60.0,
    top_k: int = 50,
) -> dict:
    with log_stage(
        logger,
        "run-combined-evaluation",
        limit=limit,
        timeout_sec=timeout_sec,
        fingerprint_url=FINGERPRINT_SERVICE_URL,
        recognition_url=ORCHESTRATION_SERVICE_URL,
        top_k=top_k,
    ):
        token = get_orchestration_token(
            ORCHESTRATION_SERVICE_URL,
            EVALUATION_USERNAME,
            EVALUATION_PASSWORD,
            timeout_sec,
        )
        queries = _load_queries(limit)
        rows = []

        fp_matched = 0
        fp_correct = 0
        recognition_matched = 0
        recognition_correct = 0
        failed = 0

        started = time.perf_counter()

        for idx, query in enumerate(queries, start=1):
            base = _base_row(query)

            try:
                fp_raw, fp_latency = retrieve_fingerprint_candidates(
                    FINGERPRINT_SERVICE_URL,
                    query["query_path"],
                    timeout_sec,
                    top_k,
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
                recognition_raw, recognition_latency = recognize_orchestration(
                    ORCHESTRATION_SERVICE_URL,
                    query["query_path"],
                    timeout_sec,
                    token,
                )
                recognition = _normalize_recognition_response(recognition_raw)
            except Exception as exc:
                failed += 1
                recognition_latency = None
                recognition = _failed_prediction(exc)
                recognition["source"] = "exception"
                logger.warning(
                    "Combined recognition query failed idx=%s query_id=%s error=%s",
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

            recognition_correct_top1 = _correct_top1(
                recognition,
                base["ground_truth_track_id"],
                base["is_positive"],
            )
            if fp["matched"]:
                fp_matched += 1
            if fp_correct_top1:
                fp_correct += 1

            if recognition["matched"]:
                recognition_matched += 1
            if recognition_correct_top1:
                recognition_correct += 1

            fp_pred = fp["predicted_track_id"]
            recognition_pred = recognition["predicted_track_id"]
            recognition_source = recognition.get("source") or "unknown"

            rows.append(
                {
                    **base,
                    "fingerprint": {
                        **fp,
                        "latency_ms": fp_latency,
                        "correct_top1": fp_correct_top1,
                        "correct_top5": fp_correct_top5,
                    },
                    "recognition": {
                        **recognition,
                        "latency_ms": recognition_latency,
                        "correct_top1": recognition_correct_top1,
                    },
                    "agreement": {
                        "same_prediction": fp_pred is not None and fp_pred == recognition_pred,
                        "both_correct": fp_correct_top1 and recognition_correct_top1,
                        "fingerprint_only_correct": fp_correct_top1 and not recognition_correct_top1,
                        "recognition_only_correct": recognition_correct_top1 and not fp_correct_top1,
                        "reranker_helped": recognition_source == "ml-reranker" and recognition_correct_top1 and not fp_correct_top1,
                        "reranker_hurt": recognition_source == "ml-reranker" and fp_correct_top1 and not recognition_correct_top1,
                        "both_wrong": base["is_positive"] and not fp_correct_top1 and not recognition_correct_top1,
                        "both_abstained": not fp["matched"] and not recognition["matched"],
                    },
                }
            )

            if idx % RUNNER_PROGRESS_EVERY == 0 or idx == len(queries):
                elapsed = time.perf_counter() - started
                logger.info(
                    "Combined eval progress queries=%s/%s fp_matched=%s fp_correct=%s recognition_matched=%s recognition_correct=%s failed=%s elapsed_sec=%.2f",
                    idx,
                    len(queries),
                    fp_matched,
                    fp_correct,
                    recognition_matched,
                    recognition_correct,
                    failed,
                    elapsed,
                )

        write_jsonl(COMBINED_RESULTS_PATH, rows)

        summary = {
            "queries": len(rows),
            "fingerprint_matched": fp_matched,
            "fingerprint_correct_top1": fp_correct,
            "recognition_matched": recognition_matched,
            "recognition_correct_top1": recognition_correct,
            "failed": failed,
            "results_path": str(COMBINED_RESULTS_PATH),
            "top_k": top_k,
        }

        logger.info("Combined evaluation finished summary=%s", summary)
        return summary
