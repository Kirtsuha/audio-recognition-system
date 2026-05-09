import json
from pathlib import Path

from pipeline.config import THRESHOLD_TUNING_PATH
from evaluation.eval_utils import read_jsonl


def _accepted(row: dict, min_confidence: float, min_margin: float, min_supported_windows: int) -> bool:
    confidence = row.get("confidence")
    margin = row.get("margin")
    support = row.get("support")

    if confidence is None or margin is None or support is None:
        return False

    return (
        float(confidence) >= min_confidence
        and float(margin) >= min_margin
        and int(support) >= min_supported_windows
    )


def tune_thresholds(
    confidence_values: list[float],
    margin_values: list[float],
    support_values: list[int],
    results_path: str | None = None,
    output_path: str | None = None,
    case: str | None = None,
    aggregation_strategy: str | None = None,
) -> dict:
    from pipeline.config import EVAL_RESULTS_PATH

    rows = read_jsonl(results_path or EVAL_RESULTS_PATH)

    if case is not None:
        rows = [x for x in rows if x.get("case") == case]

    if aggregation_strategy is not None:
        rows = [x for x in rows if x.get("aggregation_strategy") == aggregation_strategy]

    if not rows:
        raise ValueError("Eval results file is empty after filtering")

    positives = [x for x in rows if x.get("is_positive", True)]
    negatives = [x for x in rows if not x.get("is_positive", True)]

    candidates = []

    for conf in confidence_values:
        for margin in margin_values:
            for support in support_values:
                pos_accept = [x for x in positives if _accepted(x, conf, margin, support)]
                neg_accept = [x for x in negatives if _accepted(x, conf, margin, support)]

                accepted_correct_top1 = sum(1 for x in pos_accept if x.get("correct_top1"))
                accepted_correct_top5 = sum(1 for x in pos_accept if x.get("correct_top5"))

                recall_at_1 = (
                    accepted_correct_top1 / len(positives)
                    if positives else None
                )

                recall_at_5 = (
                    accepted_correct_top5 / len(positives)
                    if positives else None
                )

                precision_at_1_when_accepted = (
                    accepted_correct_top1 / len(pos_accept)
                    if pos_accept else None
                )

                accepted_rate = (
                    len(pos_accept) / len(positives)
                    if positives else None
                )

                abstain_rate = (
                    1.0 - accepted_rate
                    if accepted_rate is not None else None
                )

                false_positive_rate = (
                    len(neg_accept) / len(negatives)
                    if negatives else 0.0
                )

                objective = (
                    3.0 * (recall_at_1 or 0.0)
                    + 1.0 * (recall_at_5 or 0.0)
                    + 1.0 * (precision_at_1_when_accepted or 0.0)
                    - 4.0 * (false_positive_rate or 0.0)
                    - 0.5 * (abstain_rate or 0.0)
                )

                candidates.append(
                    {
                        "min_confidence": conf,
                        "min_margin": margin,
                        "min_supported_windows": support,
                        "recall_at_1": recall_at_1,
                        "recall_at_5": recall_at_5,
                        "precision_at_1_when_accepted": precision_at_1_when_accepted,
                        "accepted_rate": accepted_rate,
                        "abstain_rate": abstain_rate,
                        "false_positive_rate": false_positive_rate,
                        "accepted_count": len(pos_accept),
                        "total_positives": len(positives),
                        "objective": objective,
                    }
                )

    candidates.sort(key=lambda x: x["objective"], reverse=True)

    summary = {
        "filters": {
            "case": case,
            "aggregation_strategy": aggregation_strategy,
        },
        "results_path": results_path or str(EVAL_RESULTS_PATH),
        "best": candidates[0] if candidates else None,
        "top_candidates": candidates[:20],
        "total_candidates": len(candidates),
        "rows": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
    }

    if output_path is None:
        output_path = str(THRESHOLD_TUNING_PATH)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary