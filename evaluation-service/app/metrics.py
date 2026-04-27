from collections import defaultdict

from app.config import (
    COMBINED_RESULTS_PATH,
    FINGERPRINT_RESULTS_PATH,
    METRICS_PATH,
    ML_RESULTS_PATH,
)
from app.utils import percentile, read_jsonl, write_json


def _mean(values: list[float]) -> float | None:
    values = [float(x) for x in values if x is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _single_service_metrics(rows: list[dict]) -> dict:
    positives = [x for x in rows if x.get("is_positive")]
    negatives = [x for x in rows if not x.get("is_positive")]

    correct_top1 = sum(1 for x in positives if x.get("correct_top1"))
    correct_top5 = sum(1 for x in positives if x.get("correct_top5"))
    matched_pos = sum(1 for x in positives if x.get("matched"))
    false_positive = sum(1 for x in negatives if x.get("matched"))

    latencies = [x.get("latency_ms") for x in rows if x.get("latency_ms") is not None]
    confidences = [x.get("confidence") for x in rows if x.get("confidence") is not None]

    return {
        "queries": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "accuracy_at_1": correct_top1 / len(positives) if positives else None,
        "recall_at_5": correct_top5 / len(positives) if positives else None,
        "match_rate": matched_pos / len(positives) if positives else None,
        "abstain_rate": (len(positives) - matched_pos) / len(positives) if positives else None,
        "false_positive_rate": false_positive / len(negatives) if negatives else None,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "avg_confidence": _mean(confidences),
    }


def _service_from_combined(rows: list[dict], service: str) -> dict:
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
            }
        )

    return _single_service_metrics(converted)


def _comparison_metrics(rows: list[dict]) -> dict:
    positives = [x for x in rows if x.get("is_positive")]

    def count_agreement(name: str, source_rows: list[dict]) -> int:
        return sum(1 for x in source_rows if x.get("agreement", {}).get(name))

    latency_ratios = []
    for row in rows:
        fp_latency = row.get("fingerprint", {}).get("latency_ms")
        ml_latency = row.get("ml", {}).get("latency_ms")

        if fp_latency is not None and ml_latency is not None and fp_latency > 0:
            latency_ratios.append(float(ml_latency) / float(fp_latency))

    return {
        "same_prediction_rate": count_agreement("same_prediction", rows) / len(rows) if rows else None,
        "both_correct_rate": count_agreement("both_correct", positives) / len(positives) if positives else None,
        "fingerprint_only_correct_rate": count_agreement("fingerprint_only_correct", positives) / len(positives) if positives else None,
        "ml_only_correct_rate": count_agreement("ml_only_correct", positives) / len(positives) if positives else None,
        "both_wrong_rate": count_agreement("both_wrong", positives) / len(positives) if positives else None,
        "both_abstained_rate": count_agreement("both_abstained", rows) / len(rows) if rows else None,
        "avg_ml_to_fingerprint_latency_ratio": _mean(latency_ratios),
    }


def _grouped(rows: list[dict], metric_fn) -> dict:
    by_bucket = defaultdict(list)
    by_corruption = defaultdict(list)
    by_duration = defaultdict(list)

    for row in rows:
        by_bucket[str(row.get("bucket"))].append(row)
        by_corruption[str(row.get("corruption"))].append(row)
        by_duration[str(row.get("duration_sec"))].append(row)

    return {
        "overall": metric_fn(rows),
        "by_bucket": {k: metric_fn(v) for k, v in sorted(by_bucket.items())},
        "by_corruption": {k: metric_fn(v) for k, v in sorted(by_corruption.items())},
        "by_duration_sec": {
            k: metric_fn(v)
            for k, v in sorted(by_duration.items(), key=lambda x: float(x[0]))
        },
    }


def build_all_metrics() -> dict:
    fp_rows = read_jsonl(FINGERPRINT_RESULTS_PATH)
    ml_rows = read_jsonl(ML_RESULTS_PATH)
    combined_rows = read_jsonl(COMBINED_RESULTS_PATH)

    summary = {
        "fingerprint": _grouped(fp_rows, _single_service_metrics) if fp_rows else None,
        "ml": _grouped(ml_rows, _single_service_metrics) if ml_rows else None,
        "combined": None,
    }

    if combined_rows:
        summary["combined"] = {
            "overall": {
                "fingerprint": _service_from_combined(combined_rows, "fingerprint"),
                "ml": _service_from_combined(combined_rows, "ml"),
                "comparison": _comparison_metrics(combined_rows),
            }
        }

    write_json(METRICS_PATH, summary)
    return summary