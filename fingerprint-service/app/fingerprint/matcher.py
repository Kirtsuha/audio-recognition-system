from collections import defaultdict
from dataclasses import dataclass

from repository.models import Fingerprint
from config.config import (
    MIN_ALIGNED_MATCHES,
    MIN_QUERY_COVERAGE,
    MIN_SCORE_GAP,
    OFFSET_TOLERANCE_FRAMES,
    OFFSET_BIN, HOP_LENGTH, SAMPLE_RATE,
)

@dataclass
class MatchDecision:
    matched: bool
    track_id: int | None = None
    offset: int | None = None
    offset_sec: float | None = None
    aligned_matches: int = 0
    second_aligned_matches: int = 0
    query_hashes: int = 0
    unique_query_hashes: int = 0
    confidence: float = 0.0
    score_gap: float = 0.0
    reason: str | None = None
    top_candidates: list[dict] | None = None


def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def _aggregate_offset_votes(votes: dict) -> dict:
    by_track = defaultdict(list)

    for (track_id, offset), count in votes.items():
        by_track[track_id].append((offset, count))

    aggregated = {}

    for track_id, items in by_track.items():
        items = sorted(items)

        for offset, _count in items:
            total = 0

            for other_offset, other_count in items:
                if abs(other_offset - offset) <= OFFSET_TOLERANCE_FRAMES:
                    total += other_count

            key = (track_id, offset)
            aggregated[key] = max(aggregated.get(key, 0), total)

    return aggregated

def _confidence(best: int, second: int, query_hashes: int, unique_query_hashes: int) -> float:
    if query_hashes <= 0:
        return 0.0

    coverage = best / max(1, query_hashes)
    uniqueness = best / max(1, unique_query_hashes)
    separation = best / max(1, second)

    score = (
        0.50 * min(1.0, coverage / 0.08)
        + 0.30 * min(1.0, uniqueness / 0.12)
        + 0.20 * min(1.0, separation / 3.0)
    )
    return round(float(max(0.0, min(1.0, score))), 4)


def _collect_votes(hashes, db) -> tuple[dict, int, int]:
    query_offsets_by_hash = defaultdict(list)
    for h, t in hashes:
        query_offsets_by_hash[int(h)].append(int(t))

    hash_values = list(query_offsets_by_hash.keys())
    votes = defaultdict(int)

    for chunk in chunks(hash_values, 1000):
        rows = (
            db.query(Fingerprint.hash, Fingerprint.track_id, Fingerprint.time_offset)
            .filter(Fingerprint.hash.in_(chunk))
            .all()
        )

        for h, track_id, db_offset in rows:
            for query_offset in query_offsets_by_hash[int(h)]:
                delta = int(db_offset) - int(query_offset)
                delta_q = delta // OFFSET_BIN
                votes[(int(track_id), int(delta_q))] += 1

    return votes, len(hashes), len(hash_values)


def retrieve_candidates(hashes, db, top_k: int = 50) -> dict:
    if not hashes:
        return {
            "query_hashes": 0,
            "unique_query_hashes": 0,
            "candidates": [],
            "reason": "no_hashes",
        }

    votes, query_hashes, unique_query_hashes = _collect_votes(hashes, db)

    if not votes:
        return {
            "query_hashes": query_hashes,
            "unique_query_hashes": unique_query_hashes,
            "candidates": [],
            "reason": "no_votes",
        }

    votes = _aggregate_offset_votes(votes)

    by_track = defaultdict(list)

    for (track_id, offset), count in votes.items():
        by_track[int(track_id)].append(
            {
                "offset": int(offset),
                "aligned_matches": int(count),
            }
        )

    raw_candidates = []

    for track_id, offsets in by_track.items():
        offsets = sorted(
            offsets,
            key=lambda x: x["aligned_matches"],
            reverse=True,
        )

        best = offsets[0]
        total_matches = sum(x["aligned_matches"] for x in offsets)
        offset_count = len(offsets)

        best_offset = int(best["offset"])

        raw_candidates.append(
            {
                "track_id": int(track_id),
                "best_offset": best_offset,
                "best_offset_sec": offset_bin_to_seconds(best_offset),
                "aligned_matches": int(best["aligned_matches"]),
                "total_matches": int(total_matches),
                "offset_count": int(offset_count),
                "coverage": float(best["aligned_matches"] / max(1, query_hashes)),
                "unique_coverage": float(best["aligned_matches"] / max(1, unique_query_hashes)),
                "top_offsets": [
                    {
                        **x,
                        "offset_sec": offset_bin_to_seconds(x["offset"]),
                    }
                    for x in offsets[:5]
                ],
            }
        )

    raw_candidates.sort(
        key=lambda x: (
            x["aligned_matches"],
            x["total_matches"],
            x["coverage"],
        ),
        reverse=True,
    )

    top = raw_candidates[:top_k]

    best_score = top[0]["aligned_matches"] if top else 0
    second_score = top[1]["aligned_matches"] if len(top) > 1 else 0

    for candidate in top:
        candidate["score_gap"] = round(
            float(candidate["aligned_matches"] / max(1, second_score)),
            4,
        )
        candidate["confidence"] = _confidence(
            candidate["aligned_matches"],
            second_score,
            query_hashes,
            unique_query_hashes,
        )

    return {
        "query_hashes": query_hashes,
        "unique_query_hashes": unique_query_hashes,
        "top_k": top_k,
        "candidates": top,
        "reason": None if top else "no_candidates",
        "best_aligned_matches": int(best_score),
        "second_aligned_matches": int(second_score),
    }

def offset_bin_to_seconds(offset_bin: int | float) -> float:
    frame_offset = float(offset_bin) * float(OFFSET_BIN)
    seconds = frame_offset * float(HOP_LENGTH) / float(SAMPLE_RATE)
    return round(seconds, 6)

def match(hashes, db) -> dict:
    retrieved = retrieve_candidates(hashes, db, top_k=5)

    candidates = retrieved.get("candidates") or []

    if not candidates:
        return MatchDecision(
            matched=False,
            query_hashes=retrieved.get("query_hashes", 0),
            unique_query_hashes=retrieved.get("unique_query_hashes", 0),
            reason=retrieved.get("reason"),
        ).__dict__

    best_candidate = candidates[0]
    second = retrieved.get("second_aligned_matches", 0)

    track_id = int(best_candidate["track_id"])
    offset = int(best_candidate["best_offset"])
    offset_sec = float(best_candidate.get("best_offset_sec", offset_bin_to_seconds(offset)))
    best = int(best_candidate["aligned_matches"])
    score_gap = best / max(1, second)

    confidence = _confidence(
        best,
        second,
        retrieved["query_hashes"],
        retrieved["unique_query_hashes"],
    )

    accepted = (
        best >= MIN_ALIGNED_MATCHES
        and (best / max(1, retrieved["query_hashes"])) >= MIN_QUERY_COVERAGE
        and score_gap >= MIN_SCORE_GAP
    )

    return MatchDecision(
        matched=accepted,
        track_id=track_id,
        offset=offset,
        offset_sec=offset_sec,
        aligned_matches=best,
        second_aligned_matches=second,
        query_hashes=retrieved["query_hashes"],
        unique_query_hashes=retrieved["unique_query_hashes"],
        confidence=confidence,
        score_gap=round(float(score_gap), 4),
        reason=None if accepted else "low_confidence",
        top_candidates=[
            {
                "track_id": int(x["track_id"]),
                "aligned_matches": int(x["aligned_matches"]),
                "total_matches": int(x["total_matches"]),
                "best_offset": int(x["best_offset"]),
                "best_offset_sec": float(x.get("best_offset_sec", offset_bin_to_seconds(x["best_offset"]))),
                "coverage": x["coverage"],
                "confidence": x["confidence"],
            }
            for x in candidates
        ],
    ).__dict__