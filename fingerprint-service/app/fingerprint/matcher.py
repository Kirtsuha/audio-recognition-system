from collections import defaultdict
from dataclasses import dataclass

from repository.models import Fingerprint
from config.config import MIN_ALIGNED_MATCHES, MIN_QUERY_COVERAGE, MIN_SCORE_GAP


@dataclass
class MatchDecision:
    matched: bool
    track_id: int | None = None
    offset: int | None = None
    aligned_matches: int = 0
    second_aligned_matches: int = 0
    query_hashes: int = 0
    unique_query_hashes: int = 0
    confidence: float = 0.0
    score_gap: float = 0.0
    reason: str | None = None


def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def _confidence(best: int, second: int, query_hashes: int, unique_query_hashes: int) -> float:
    if query_hashes <= 0:
        return 0.0

    coverage = best / max(1, query_hashes)
    uniqueness = best / max(1, unique_query_hashes)
    separation = best / max(1, second)

    # Консервативная эвристика.
    score = (
        0.50 * min(1.0, coverage / 0.08)
        + 0.30 * min(1.0, uniqueness / 0.12)
        + 0.20 * min(1.0, separation / 3.0)
    )
    return round(float(max(0.0, min(1.0, score))), 4)


def match(hashes, db) -> dict:
    if not hashes:
        return MatchDecision(matched=False, reason="no_hashes").__dict__

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
                votes[(int(track_id), delta)] += 1

    if not votes:
        return MatchDecision(
            matched=False,
            query_hashes=len(hashes),
            unique_query_hashes=len(hash_values),
            reason="no_votes",
        ).__dict__

    ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    (track_id, offset), best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0

    score_gap = best / max(1, second)
    confidence = _confidence(best, second, len(hashes), len(hash_values))

    accepted = (
        best >= MIN_ALIGNED_MATCHES
        and (best / max(1, len(hashes))) >= MIN_QUERY_COVERAGE
        and score_gap >= MIN_SCORE_GAP
    )

    if not accepted:
        return MatchDecision(
            matched=False,
            track_id=track_id,
            offset=offset,
            aligned_matches=best,
            second_aligned_matches=second,
            query_hashes=len(hashes),
            unique_query_hashes=len(hash_values),
            confidence=confidence,
            score_gap=round(float(score_gap), 4),
            reason="low_confidence",
        ).__dict__

    return MatchDecision(
        matched=True,
        track_id=track_id,
        offset=offset,
        aligned_matches=best,
        second_aligned_matches=second,
        query_hashes=len(hashes),
        unique_query_hashes=len(hash_values),
        confidence=confidence,
        score_gap=round(float(score_gap), 4),
    ).__dict__