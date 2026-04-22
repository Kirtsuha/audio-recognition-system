from collections import defaultdict
from typing import Any

import numpy as np


def aggregate_results(
    scores: np.ndarray,
    indices: np.ndarray,
    song_ids: np.ndarray,
) -> dict[str, Any] | None:
    if scores.size == 0 or indices.size == 0:
        return None

    stats: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "sum_score": 0.0,
            "max_score": float("-inf"),
            "hit_count": 0,
            "supported_queries": set(),
            "rank_bonus": 0.0,
        }
    )

    num_queries = indices.shape[0]

    for q_idx in range(num_queries):
        for rank, (score, emb_idx) in enumerate(zip(scores[q_idx], indices[q_idx]), start=1):
            if emb_idx < 0:
                continue

            song_id = int(song_ids[emb_idx])
            rank_weight = 1.0 / rank

            stats[song_id]["sum_score"] += float(score)
            stats[song_id]["max_score"] = max(stats[song_id]["max_score"], float(score))
            stats[song_id]["hit_count"] += 1
            stats[song_id]["supported_queries"].add(q_idx)
            stats[song_id]["rank_bonus"] += rank_weight

    if not stats:
        return None

    candidates = []
    for song_id, item in stats.items():
        support = len(item["supported_queries"])
        support_ratio = support / max(num_queries, 1)

        final_score = (
            item["sum_score"]
            + 0.35 * item["rank_bonus"]
            + 0.30 * support
            + 0.15 * item["max_score"]
            + 0.20 * support_ratio
        )

        candidates.append(
            {
                "song_id": song_id,
                "final_score": float(final_score),
                "sum_score": float(item["sum_score"]),
                "max_score": float(item["max_score"]),
                "hit_count": int(item["hit_count"]),
                "support": int(support),
                "support_ratio": float(support_ratio),
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
    }