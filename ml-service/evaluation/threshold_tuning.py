import json

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
) -> dict:
    from pipeline.config import EVAL_RESULTS_PATH

    rows = read_jsonl(results_path or EVAL_RESULTS_PATH)
    if not rows:
        raise ValueError("Eval results file is empty")

    positives = [x for x in rows if x["is_positive"]]
    negatives = [x for x in rows if not x["is_positive"]]

    candidates = []

    for conf in confidence_values:
        for margin in margin_values:
            for support in support_values:
                pos_accept = [x for x in positives if _accepted(x, conf, margin, support)]
                neg_accept = [x for x in negatives if _accepted(x, conf, margin, support)]

                accuracy_at_1 = (
                    sum(1 for x in pos_accept if x["correct_top1"]) / len(positives)
                    if positives else None
                )
                recall_at_5 = (
                    sum(1 for x in pos_accept if x["correct_top5"]) / len(positives)
                    if positives else None
                )
                abstain_rate = (
                    sum(1 for x in positives if not _accepted(x, conf, margin, support)) / len(positives)
                    if positives else None
                )
                false_positive_rate = (
                    len(neg_accept) / len(negatives)
                    if negatives else None
                )

                # Conservative score: reward accuracy, punish false positives and abstains
                objective = (
                    3.0 * (accuracy_at_1 or 0.0)
                    + 1.0 * (recall_at_5 or 0.0)
                    - 4.0 * (false_positive_rate or 0.0)
                    - 0.5 * (abstain_rate or 0.0)
                )

                candidates.append(
                    {
                        "min_confidence": conf,
                        "min_margin": margin,
                        "min_supported_windows": support,
                        "accuracy_at_1": accuracy_at_1,
                        "recall_at_5": recall_at_5,
                        "abstain_rate": abstain_rate,
                        "false_positive_rate": false_positive_rate,
                        "objective": objective,
                    }
                )

    candidates.sort(key=lambda x: x["objective"], reverse=True)

    summary = {
        "best": candidates[0] if candidates else None,
        "top_candidates": candidates[:20],
        "total_candidates": len(candidates),
    }

    THRESHOLD_TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THRESHOLD_TUNING_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary