from collections import defaultdict

from fingerprint_experiment.utils import percentile


def _mean(values: list[float]) -> float | None:
    values = [float(x) for x in values if x is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _group_metrics(rows: list[dict]) -> dict:
    positives = [x for x in rows if x["is_positive"]]
    negatives = [x for x in rows if not x["is_positive"]]

    correct_top1 = sum(1 for x in positives if x["correct_top1"])
    correct_top5 = sum(1 for x in positives if x["correct_top5"])
    matched_pos = sum(1 for x in positives if x["matched"])
    false_positive = sum(1 for x in negatives if x["matched"])

    return {
        "queries": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "accuracy_at_1": correct_top1 / len(positives) if positives else None,
        "recall_at_5": correct_top5 / len(positives) if positives else None,
        "match_rate": matched_pos / len(positives) if positives else None,
        "abstain_rate": (len(positives) - matched_pos) / len(positives) if positives else None,
        "false_positive_rate": false_positive / len(negatives) if negatives else None,
        "latency_p50_ms": percentile([x.get("latency_ms") for x in rows], 0.50),
        "latency_p95_ms": percentile([x.get("latency_ms") for x in rows], 0.95),
        "latency_p99_ms": percentile([x.get("latency_ms") for x in rows], 0.99),
        "avg_confidence": _mean([x.get("confidence") for x in rows]),
        "avg_query_hashes": _mean([x.get("query_hashes") for x in rows]),
        "avg_aligned_matches": _mean([x.get("aligned_matches") for x in rows]),
    }


def build_metrics(rows: list[dict]) -> dict:
    by_bucket = defaultdict(list)
    by_corruption = defaultdict(list)
    by_duration = defaultdict(list)

    for row in rows:
        by_bucket[str(row.get("bucket"))].append(row)
        by_corruption[str(row.get("corruption"))].append(row)
        by_duration[str(row.get("duration_sec"))].append(row)

    return {
        "overall": _group_metrics(rows),
        "by_bucket": {k: _group_metrics(v) for k, v in sorted(by_bucket.items())},
        "by_corruption": {k: _group_metrics(v) for k, v in sorted(by_corruption.items())},
        "by_duration_sec": {
            k: _group_metrics(v)
            for k, v in sorted(by_duration.items(), key=lambda x: float(x[0]))
        },
    }