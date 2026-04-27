from app.clients import recognize_fingerprint, recognize_ml
from app.config import (
    COMBINED_RESULTS_PATH,
    EVAL_QUERIES_PATH,
    FINGERPRINT_RESULTS_PATH,
    FINGERPRINT_SERVICE_URL,
    ML_RESULTS_PATH,
    ML_SERVICE_URL,
)
from app.utils import read_jsonl, safe_int, write_jsonl


def _normalize_fingerprint_response(response: dict) -> dict:
    return {
        "matched": bool(response.get("match", response.get("matched", False))),
        "predicted_track_id": safe_int(response.get("track_id") or response.get("trackId")),
        "confidence": response.get("confidence"),
        "reason": response.get("reason"),
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


def _correct_top5_ml(prediction: dict, gt_track_id: int | None, is_positive: bool) -> bool:
    if not is_positive or gt_track_id is None:
        return False

    ids = []
    for candidate in prediction.get("top_candidates", [])[:5]:
        ids.append(
            safe_int(
                candidate.get("song_id")
                or candidate.get("track_id")
                or candidate.get("trackId")
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
    }


def run_fingerprint_evaluation(
    limit: int | None = None,
    timeout_sec: float = 60.0,
) -> dict:
    queries = read_jsonl(EVAL_QUERIES_PATH)
    if limit is not None:
        queries = queries[:limit]

    rows = []

    for query in queries:
        base = _base_row(query)

        try:
            raw, latency_ms = recognize_fingerprint(
                FINGERPRINT_SERVICE_URL,
                query["query_path"],
                timeout_sec,
            )
            pred = _normalize_fingerprint_response(raw)
        except Exception as exc:
            latency_ms = None
            pred = _failed_prediction(exc)

        correct_top1 = _correct_top1(
            pred,
            base["ground_truth_track_id"],
            base["is_positive"],
        )

        rows.append(
            {
                **base,
                "predicted_track_id": pred["predicted_track_id"],
                "matched": pred["matched"],
                "correct_top1": correct_top1,
                "correct_top5": correct_top1,
                "confidence": pred.get("confidence"),
                "reason": pred.get("reason"),
                "latency_ms": latency_ms,
                "raw_prediction": pred.get("raw_response"),
            }
        )

    write_jsonl(FINGERPRINT_RESULTS_PATH, rows)

    return {
        "queries": len(rows),
        "results_path": str(FINGERPRINT_RESULTS_PATH),
    }


def run_ml_evaluation(
    limit: int | None = None,
    timeout_sec: float = 60.0,
) -> dict:
    queries = read_jsonl(EVAL_QUERIES_PATH)
    if limit is not None:
        queries = queries[:limit]

    rows = []

    for query in queries:
        base = _base_row(query)

        try:
            raw, latency_ms = recognize_ml(
                ML_SERVICE_URL,
                query["query_path"],
                timeout_sec,
            )
            pred = _normalize_ml_response(raw)
        except Exception as exc:
            latency_ms = None
            pred = _failed_prediction(exc)

        correct_top1 = _correct_top1(
            pred,
            base["ground_truth_track_id"],
            base["is_positive"],
        )
        correct_top5 = _correct_top5_ml(
            pred,
            base["ground_truth_track_id"],
            base["is_positive"],
        )

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

    write_jsonl(ML_RESULTS_PATH, rows)

    return {
        "queries": len(rows),
        "results_path": str(ML_RESULTS_PATH),
    }


def run_combined_evaluation(
    limit: int | None = None,
    timeout_sec: float = 60.0,
) -> dict:
    queries = read_jsonl(EVAL_QUERIES_PATH)
    if limit is not None:
        queries = queries[:limit]

    rows = []

    for query in queries:
        base = _base_row(query)

        try:
            fp_raw, fp_latency = recognize_fingerprint(
                FINGERPRINT_SERVICE_URL,
                query["query_path"],
                timeout_sec,
            )
            fp = _normalize_fingerprint_response(fp_raw)
        except Exception as exc:
            fp_latency = None
            fp = _failed_prediction(exc)

        try:
            ml_raw, ml_latency = recognize_ml(
                ML_SERVICE_URL,
                query["query_path"],
                timeout_sec,
            )
            ml = _normalize_ml_response(ml_raw)
        except Exception as exc:
            ml_latency = None
            ml = _failed_prediction(exc)

        fp_correct_top1 = _correct_top1(
            fp,
            base["ground_truth_track_id"],
            base["is_positive"],
        )

        ml_correct_top1 = _correct_top1(
            ml,
            base["ground_truth_track_id"],
            base["is_positive"],
        )

        ml_correct_top5 = _correct_top5_ml(
            ml,
            base["ground_truth_track_id"],
            base["is_positive"],
        )

        fp_pred = fp["predicted_track_id"]
        ml_pred = ml["predicted_track_id"]

        rows.append(
            {
                **base,
                "fingerprint": {
                    **fp,
                    "latency_ms": fp_latency,
                    "correct_top1": fp_correct_top1,
                    "correct_top5": fp_correct_top1,
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

    write_jsonl(COMBINED_RESULTS_PATH, rows)

    return {
        "queries": len(rows),
        "results_path": str(COMBINED_RESULTS_PATH),
    }