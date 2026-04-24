import json
from collections import defaultdict

from pipeline.config import EVAL_METRICS_PATH, EVAL_RESULTS_PATH
from evaluation.eval_utils import percentile, read_jsonl


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _compute_group_metrics(rows: list[dict]) -> dict:
    positives = [x for x in rows if x["is_positive"]]
    negatives = [x for x in rows if not x["is_positive"]]

    latencies = [float(x["latency_ms"]) for x in rows if x.get("latency_ms") is not None]
    pos_confidences = [float(x["confidence"]) for x in positives if x.get("confidence") is not None]
    pos_margins = [float(x["margin"]) for x in positives if x.get("margin") is not None]

    correct_top1 = sum(1 for x in positives if x["correct_top1"])
    correct_top5 = sum(1 for x in positives if x["correct_top5"])
    matched_pos = sum(1 for x in positives if x["matched"])
    abstained_pos = sum(1 for x in positives if not x["matched"])
    false_positive = sum(1 for x in negatives if x["matched"])

    return {
        "queries": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "accuracy_at_1": correct_top1 / len(positives) if positives else None,
        "recall_at_5": correct_top5 / len(positives) if positives else None,
        "match_rate": matched_pos / len(positives) if positives else None,
        "abstain_rate": abstained_pos / len(positives) if positives else None,
        "false_positive_rate": false_positive / len(negatives) if negatives else None,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "avg_confidence_positive": _safe_mean(pos_confidences),
        "avg_margin_positive": _safe_mean(pos_margins),
    }


def build_eval_metrics() -> dict:
    rows = read_jsonl(EVAL_RESULTS_PATH)
    if not rows:
        raise ValueError("Eval results file is empty")

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    by_corruption: dict[str, list[dict]] = defaultdict(list)
    by_duration: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        by_bucket[row["bucket"]].append(row)
        by_corruption[row["corruption"]].append(row)
        by_duration[str(row["duration_sec"])].append(row)

    summary = {
        "overall": _compute_group_metrics(rows),
        "by_bucket": {k: _compute_group_metrics(v) for k, v in sorted(by_bucket.items())},
        "by_corruption": {k: _compute_group_metrics(v) for k, v in sorted(by_corruption.items())},
        "by_duration_sec": {k: _compute_group_metrics(v) for k, v in sorted(by_duration.items(), key=lambda x: float(x[0]))},
    }

    EVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary