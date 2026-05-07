import logging
from collections import defaultdict

from app.config import (
    COMBINED_RESULTS_PATH,
    FINGERPRINT_RESULTS_PATH,
    METRICS_PATH,
    RECOGNITION_RESULTS_PATH,
)
from app.logging_utils import log_stage
from app.utils import percentile, read_jsonl, write_json

logger = logging.getLogger("evaluation.metrics")


def _mean(values: list[float]) -> float | None:
    values = [float(x) for x in values if x is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _single_service_metrics(
    rows: list[dict],
    include_recall_at_5: bool = True,
) -> dict:
    positives = [x for x in rows if x.get("is_positive")]
    negatives = [x for x in rows if not x.get("is_positive")]

    correct_top1 = sum(1 for x in positives if x.get("correct_top1"))
    correct_top5 = sum(1 for x in positives if x.get("correct_top5"))
    matched_pos = sum(1 for x in positives if x.get("matched"))
    false_positive = sum(1 for x in negatives if x.get("matched"))

    latencies = [x.get("latency_ms") for x in rows if x.get("latency_ms") is not None]
    confidences = [x.get("confidence") for x in rows if x.get("confidence") is not None]

    result = {
        "queries": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "accuracy": correct_top1 / len(positives) if positives else None,
        "accuracy_at_1": correct_top1 / len(positives) if positives else None,
        "match_rate": matched_pos / len(positives) if positives else None,
        "abstain_rate": (len(positives) - matched_pos) / len(positives) if positives else None,
        "false_positive_rate": false_positive / len(negatives) if negatives else None,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "avg_confidence": _mean(confidences),
        "source_counts": _source_counts(rows),
        "source_rates": _source_rates(rows),
    }

    if include_recall_at_5:
        result["recall_at_5"] = correct_top5 / len(positives) if positives else None

    return result


def _source_counts(rows: list[dict]) -> dict:
    counts = defaultdict(int)
    for row in rows:
        source = row.get("source")
        if source is not None:
            counts[str(source)] += 1
    return dict(sorted(counts.items()))


def _source_rates(rows: list[dict]) -> dict:
    counts = _source_counts(rows)
    total = len(rows)
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _service_from_combined(
    rows: list[dict],
    service: str,
    include_recall_at_5: bool = True,
) -> dict:
    converted = []

    for row in rows:
        pred = row.get(service, {})
        converted.append(
            {
                "is_positive": row.get("is_positive"),
                "matched": pred.get("matched"),
                "correct_top1": pred.get("correct_top1"),
                "correct_top5": pred.get("correct_top5"),
                "confidence": pred.get("confidence"),
                "latency_ms": pred.get("latency_ms"),
                "source": pred.get("source"),
            }
        )

    return _single_service_metrics(
        converted,
        include_recall_at_5=include_recall_at_5,
    )


def _comparison_metrics(rows: list[dict]) -> dict:
    positives = [x for x in rows if x.get("is_positive")]

    def count_agreement(name: str, source_rows: list[dict]) -> int:
        return sum(1 for x in source_rows if x.get("agreement", {}).get(name))

    latency_ratios = []
    for row in rows:
        fp_latency = row.get("fingerprint", {}).get("latency_ms")
        recognition = row.get("recognition") or {}
        recognition_latency = recognition.get("latency_ms")

        if fp_latency is not None and recognition_latency is not None and fp_latency > 0:
            latency_ratios.append(float(recognition_latency) / float(fp_latency))

    return {
        "same_prediction_rate": count_agreement("same_prediction", rows) / len(rows) if rows else None,
        "both_correct_rate": count_agreement("both_correct", positives) / len(positives) if positives else None,
        "fingerprint_only_correct_rate": count_agreement("fingerprint_only_correct", positives) / len(positives) if positives else None,
        "recognition_only_correct_rate": count_agreement("recognition_only_correct", positives) / len(positives) if positives else None,
        "reranker_helped_rate": count_agreement("reranker_helped", positives) / len(positives) if positives else None,
        "reranker_hurt_rate": count_agreement("reranker_hurt", positives) / len(positives) if positives else None,
        "both_wrong_rate": count_agreement("both_wrong", positives) / len(positives) if positives else None,
        "both_abstained_rate": count_agreement("both_abstained", rows) / len(rows) if rows else None,
        "avg_recognition_to_fingerprint_latency_ratio": _mean(latency_ratios),
    }


def _grouped(rows: list[dict], metric_fn, **metric_kwargs) -> dict:
    by_bucket = defaultdict(list)
    by_corruption = defaultdict(list)
    by_duration = defaultdict(list)
    by_audio_source = defaultdict(list)

    for row in rows:
        by_bucket[str(row.get("bucket"))].append(row)
        by_corruption[str(row.get("corruption"))].append(row)
        by_duration[str(row.get("duration_sec"))].append(row)
        by_audio_source[str(row.get("audio_source"))].append(row)

    return {
        "overall": metric_fn(rows, **metric_kwargs),
        "by_bucket": {k: metric_fn(v, **metric_kwargs) for k, v in sorted(by_bucket.items())},
        "by_corruption": {k: metric_fn(v, **metric_kwargs) for k, v in sorted(by_corruption.items())},
        "by_duration_sec": {
            k: metric_fn(v, **metric_kwargs)
            for k, v in sorted(by_duration.items(), key=lambda x: float(x[0]))
        },
        "by_audio_source": {
            k: metric_fn(v, **metric_kwargs)
            for k, v in sorted(by_audio_source.items())
        },
    }

def _delete_if_exists(path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Failed to delete old file path=%s", path, exc_info=True)

def build_all_metrics() -> dict:
    with log_stage(logger, "build-all-metrics"):
        fp_rows = read_jsonl(FINGERPRINT_RESULTS_PATH)
        recognition_rows = read_jsonl(RECOGNITION_RESULTS_PATH)
        combined_rows = read_jsonl(COMBINED_RESULTS_PATH)

        logger.info(
            "Loaded result rows fingerprint=%s recognition=%s combined=%s",
            len(fp_rows),
            len(recognition_rows),
            len(combined_rows),
        )

        summary = {
            "fingerprint": _grouped(fp_rows, _single_service_metrics) if fp_rows else None,
            "recognition": _grouped(
                recognition_rows,
                _single_service_metrics,
                include_recall_at_5=False,
            ) if recognition_rows else None,
            "combined": None,
        }

        if combined_rows:
            summary["combined"] = {
                "overall": {
                    "fingerprint": _service_from_combined(combined_rows, "fingerprint"),
                    "recognition": _service_from_combined(
                        combined_rows,
                        "recognition",
                        include_recall_at_5=False,
                    ),
                    "comparison": _comparison_metrics(combined_rows),
                }
            }

        write_json(METRICS_PATH, summary)

        logger.info("Metrics written path=%s", METRICS_PATH)

        return summary
