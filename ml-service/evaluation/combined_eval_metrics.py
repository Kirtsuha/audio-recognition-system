import json
import os
from collections import defaultdict
from pathlib import Path

from evaluation.eval_utils import percentile, read_jsonl

COMBINED_RESULTS_PATH = os.getenv(
    "COMBINED_RESULTS_PATH",
    "/app/artifacts/eval/combined_eval_results.jsonl",
)

COMBINED_METRICS_PATH = os.getenv(
    "COMBINED_METRICS_PATH",
    "/app/artifacts/eval/combined_eval_metrics.json",
)


def _safe_mean(values: list[float]) -> float | None:
    values = [float(x) for x in values if x is not None]

    if not values:
        return None

    return float(sum(values) / len(values))


def _service_metrics(rows: list[dict], service_name: str) -> dict:
    positives = [x for x in rows if x.get("is_positive")]
    negatives = [x for x in rows if not x.get("is_positive")]

    service_rows = [x.get(service_name, {}) for x in rows]
    pos_service_rows = [x.get(service_name, {}) for x in positives]
    neg_service_rows = [x.get(service_name, {}) for x in negatives]

    correct_top1 = sum(1 for x in pos_service_rows if x.get("correct_top1"))
    correct_top5 = sum(1 for x in pos_service_rows if x.get("correct_top5"))

    matched_pos = sum(1 for x in pos_service_rows if x.get("matched"))
    false_positive = sum(1 for x in neg_service_rows if x.get("matched"))

    latencies = [
        x.get("latency_ms")
        for x in service_rows
        if x.get("latency_ms") is not None
    ]

    confidences = [
        x.get("confidence")
        for x in service_rows
        if x.get("confidence") is not None
    ]

    return {
        "accuracy_at_1": correct_top1 / len(positives) if positives else None,
        "recall_at_5": correct_top5 / len(positives) if positives else None,
        "match_rate": matched_pos / len(positives) if positives else None,
        "abstain_rate": (
            (len(positives) - matched_pos) / len(positives)
            if positives
            else None
        ),
        "false_positive_rate": (
            false_positive / len(negatives)
            if negatives
            else None
        ),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "avg_confidence": _safe_mean(confidences),
    }


def _comparison_metrics(rows: list[dict]) -> dict:
    positives = [x for x in rows if x.get("is_positive")]

    same_prediction = sum(
        1 for x in rows
        if x.get("agreement", {}).get("same_prediction")
    )

    both_correct = sum(
        1 for x in positives
        if x.get("agreement", {}).get("both_correct")
    )

    ml_only_correct = sum(
        1 for x in positives
        if x.get("agreement", {}).get("ml_only_correct")
    )

    fp_only_correct = sum(
        1 for x in positives
        if x.get("agreement", {}).get("fingerprint_only_correct")
    )

    both_wrong = sum(
        1 for x in positives
        if x.get("agreement", {}).get("both_wrong")
    )

    both_abstained = sum(
        1 for x in rows
        if x.get("agreement", {}).get("both_abstained")
    )

    latency_ratios = []

    for row in rows:
        fp_latency = row.get("fingerprint", {}).get("latency_ms")
        ml_latency = row.get("ml", {}).get("latency_ms")

        if fp_latency is not None and ml_latency is not None and fp_latency > 0:
            latency_ratios.append(float(ml_latency) / float(fp_latency))

    return {
        "same_prediction_rate": same_prediction / len(rows) if rows else None,
        "both_correct_rate": both_correct / len(positives) if positives else None,
        "ml_only_correct_rate": ml_only_correct / len(positives) if positives else None,
        "fingerprint_only_correct_rate": fp_only_correct / len(positives) if positives else None,
        "both_wrong_rate": both_wrong / len(positives) if positives else None,
        "both_abstained_rate": both_abstained / len(rows) if rows else None,
        "avg_ml_to_fingerprint_latency_ratio": _safe_mean(latency_ratios),
    }


def _group_summary(rows: list[dict]) -> dict:
    return {
        "ml": _service_metrics(rows, "ml"),
        "fingerprint": _service_metrics(rows, "fingerprint"),
        "comparison": _comparison_metrics(rows),
    }


def build_combined_eval_metrics(
    results_path: str | None = None,
    metrics_path: str | None = None,
) -> dict:
    results_path = results_path or COMBINED_RESULTS_PATH
    metrics_path = metrics_path or COMBINED_METRICS_PATH

    rows = read_jsonl(results_path)

    if not rows:
        raise ValueError(f"Combined eval results file is empty: {results_path}")

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    by_corruption: dict[str, list[dict]] = defaultdict(list)
    by_duration: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        by_bucket[str(row.get("bucket"))].append(row)
        by_corruption[str(row.get("corruption"))].append(row)
        by_duration[str(row.get("duration_sec"))].append(row)

    summary = {
        "overall": _group_summary(rows),
        "by_bucket": {
            k: _group_summary(v)
            for k, v in sorted(by_bucket.items())
        },
        "by_corruption": {
            k: _group_summary(v)
            for k, v in sorted(by_corruption.items())
        },
        "by_duration_sec": {
            k: _group_summary(v)
            for k, v in sorted(
                by_duration.items(),
                key=lambda x: float(x[0]),
            )
        },
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary