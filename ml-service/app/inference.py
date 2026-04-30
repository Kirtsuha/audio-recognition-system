from collections import defaultdict
from typing import Any

import numpy as np


def aggregate_results(
    scores: np.ndarray,
    indices: np.ndarray,
    song_ids: np.ndarray,
    strategy: str = "current",
    support_score_threshold: float = 0.50,
) -> dict[str, Any] | None:
    if scores.size == 0 or indices.size == 0:
        return None

    stats: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "scores": [],
            "sum_score": 0.0,
            "max_score": float("-inf"),
            "hit_count": 0,
            "supported_queries": set(),
            "rank_bonus": 0.0,
            "support_above_threshold": 0,
        }
    )

    num_queries = indices.shape[0]

    for q_idx in range(num_queries):
        for rank, (score, emb_idx) in enumerate(zip(scores[q_idx], indices[q_idx]), start=1):
            if emb_idx < 0:
                continue

            score = float(score)
            song_id = int(song_ids[emb_idx])
            rank_weight = 1.0 / rank

            item = stats[song_id]
            item["scores"].append(score)
            item["sum_score"] += score
            item["max_score"] = max(item["max_score"], score)
            item["hit_count"] += 1
            item["supported_queries"].add(q_idx)
            item["rank_bonus"] += rank_weight

            if score >= support_score_threshold:
                item["support_above_threshold"] += 1

    if not stats:
        return None

    candidates = []

    for song_id, item in stats.items():
        support = len(item["supported_queries"])
        support_ratio = support / max(num_queries, 1)
        scores_list = sorted(item["scores"], reverse=True)

        top3_avg = float(np.mean(scores_list[:3])) if scores_list else 0.0
        top5_avg = float(np.mean(scores_list[:5])) if scores_list else 0.0

        if strategy == "current":
            final_score = (
                item["sum_score"]
                + 0.35 * item["rank_bonus"]
                + 0.30 * support
                + 0.15 * item["max_score"]
                + 0.20 * support_ratio
            )

        elif strategy == "max":
            final_score = item["max_score"]

        elif strategy == "top3_avg":
            final_score = top3_avg

        elif strategy == "top5_avg":
            final_score = top5_avg

        elif strategy == "sum":
            final_score = item["sum_score"]

        elif strategy == "rank_bonus":
            final_score = item["rank_bonus"]

        elif strategy == "support":
            final_score = support

        elif strategy == "support_threshold":
            final_score = (
                item["support_above_threshold"]
                + 0.10 * item["max_score"]
                + 0.05 * item["rank_bonus"]
            )

        elif strategy == "support_then_max":
            final_score = (
                support * 10.0
                + item["max_score"]
                + 0.01 * item["rank_bonus"]
            )

        elif strategy == "hybrid_v2":
            final_score = (
                1.50 * top3_avg
                + 0.80 * item["max_score"]
                + 0.50 * support
                + 0.20 * item["rank_bonus"]
                + 0.30 * support_ratio
            )

        else:
            raise ValueError(f"Unsupported aggregation strategy={strategy}")

        candidates.append(
            {
                "song_id": int(song_id),
                "final_score": float(final_score),
                "sum_score": float(item["sum_score"]),
                "max_score": float(item["max_score"]),
                "top3_avg": float(top3_avg),
                "top5_avg": float(top5_avg),
                "hit_count": int(item["hit_count"]),
                "support": int(support),
                "support_ratio": float(support_ratio),
                "support_above_threshold": int(item["support_above_threshold"]),
            }
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    top1 = candidates[0]
    top2 = candidates[1] if len(candidates) > 1 else None

    second_score = top2["final_score"] if top2 else 1e-6
    margin = top1["final_score"] - second_score
    confidence = top1["final_score"] / (top1["final_score"] + second_score + 1e-8)

    return {
        "song_id": int(top1["song_id"]),
        "score": float(top1["final_score"]),
        "confidence": float(confidence),
        "margin": float(margin),
        "support": int(top1["support"]),
        "support_ratio": float(top1["support_ratio"]),
        "top_candidates": candidates[:5],
        "aggregation_strategy": strategy,
    }