from collections import defaultdict
from dataclasses import asdict, dataclass

import librosa
import numpy as np
from scipy.ndimage import maximum_filter

from fingerprint_experiment.schemas import FingerprintParams


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
    top_candidates: list[dict] | None = None


def spectrogram_db(audio: np.ndarray, params: FingerprintParams) -> np.ndarray:
    s = np.abs(
        librosa.stft(
            audio,
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            window="hann",
        )
    )
    return librosa.amplitude_to_db(s, ref=np.max)


def find_peaks(s_db: np.ndarray, params: FingerprintParams) -> list[tuple[int, int]]:
    neighborhood = maximum_filter(s_db, size=params.peak_neighborhood_size)
    mask = (s_db == neighborhood) & (s_db > params.amp_min_db)

    freq_idx, time_idx = np.where(mask)

    candidates = list(zip(time_idx, freq_idx, s_db[freq_idx, time_idx]))
    candidates.sort(key=lambda x: (x[0], -x[2]))

    result = []
    counts = {}

    for t, f, amp in candidates:
        count = counts.get(int(t), 0)
        if count >= params.max_peaks_per_frame:
            continue

        result.append((int(t), int(f)))
        counts[int(t)] = count + 1

    return result


def pack_hash(f1: int, f2: int, delta_t: int, params: FingerprintParams) -> int:
    qf1 = int(f1) // params.freq_bin_size
    qf2 = int(f2) // params.freq_bin_size
    qdt = int(round(int(delta_t) / params.delta_t_bin_size))

    return ((qf1 & 0xFFF) << 20) | ((qf2 & 0xFFF) << 8) | (qdt & 0xFF)


def generate_hashes(
    peaks: list[tuple[int, int]],
    params: FingerprintParams,
) -> list[tuple[int, int]]:
    peaks = sorted(peaks, key=lambda x: (x[0], x[1]))

    by_time = defaultdict(list)
    for t, f in peaks:
        by_time[int(t)].append(int(f))

    times = sorted(by_time.keys())
    hashes = []

    for time_idx, t1 in enumerate(times):
        future = []

        for t2 in times[time_idx + 1:]:
            dt = t2 - t1

            if dt < params.min_delta_t:
                continue

            if dt > params.max_delta_t:
                break

            for f2 in by_time[t2]:
                future.append((dt, f2))

        if not future:
            continue

        if len(future) > params.fan_value:
            step = len(future) / params.fan_value
            future = [future[int(i * step)] for i in range(params.fan_value)]

        for f1 in by_time[t1]:
            for dt, f2 in future:
                hashes.append((pack_hash(f1, f2, dt, params), t1))

    return hashes


def fingerprint_audio_array(
    audio: np.ndarray,
    params: FingerprintParams,
) -> dict:
    if audio is None or len(audio) == 0:
        return {
            "hashes": [],
            "peaks": 0,
            "frames": 0,
        }

    s_db = spectrogram_db(audio, params)
    peaks = find_peaks(s_db, params)
    hashes = generate_hashes(peaks, params)

    return {
        "hashes": hashes,
        "peaks": len(peaks),
        "frames": int(s_db.shape[1]),
    }


def build_memory_index(
    track_hashes: dict[int, list[tuple[int, int]]],
) -> dict[int, list[tuple[int, int]]]:
    index = defaultdict(list)

    for track_id, hashes in track_hashes.items():
        for h, t in hashes:
            index[int(h)].append((int(track_id), int(t)))

    return dict(index)


def _aggregate_offset_votes(
    votes: dict[tuple[int, int], int],
    params: FingerprintParams,
) -> dict[tuple[int, int], int]:
    by_track = defaultdict(list)

    for (track_id, offset), count in votes.items():
        by_track[track_id].append((offset, count))

    aggregated = {}

    for track_id, items in by_track.items():
        items = sorted(items)

        for offset, _count in items:
            total = 0

            for other_offset, other_count in items:
                if abs(other_offset - offset) <= params.offset_tolerance_bins:
                    total += other_count

            aggregated[(track_id, offset)] = max(
                aggregated.get((track_id, offset), 0),
                total,
            )

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


def match_in_memory(
    query_hashes: list[tuple[int, int]],
    index: dict[int, list[tuple[int, int]]],
    params: FingerprintParams,
) -> dict:
    if not query_hashes:
        return asdict(MatchDecision(matched=False, reason="no_hashes"))

    query_offsets_by_hash = defaultdict(list)
    for h, t in query_hashes:
        query_offsets_by_hash[int(h)].append(int(t))

    votes = defaultdict(int)

    for h, query_offsets in query_offsets_by_hash.items():
        rows = index.get(int(h), [])

        for track_id, db_offset in rows:
            for query_offset in query_offsets:
                delta = int(db_offset) - int(query_offset)
                delta_q = delta // params.offset_bin
                votes[(int(track_id), int(delta_q))] += 1

    if not votes:
        return asdict(
            MatchDecision(
                matched=False,
                query_hashes=len(query_hashes),
                unique_query_hashes=len(query_offsets_by_hash),
                reason="no_votes",
            )
        )

    votes = _aggregate_offset_votes(votes, params)

    track_scores = defaultdict(int)
    track_total_scores = defaultdict(int)

    for (track_id, _offset), count in votes.items():
        track_scores[track_id] = max(track_scores[track_id], count)
        track_total_scores[track_id] += count

    ranked_tracks = sorted(
        track_scores.items(),
        key=lambda x: (x[1], track_total_scores[x[0]]),
        reverse=True,
    )

    best_track_id, _best_score = ranked_tracks[0]
    second = ranked_tracks[1][1] if len(ranked_tracks) > 1 else 0

    best_offsets = [
        ((track_id, offset), count)
        for (track_id, offset), count in votes.items()
        if track_id == best_track_id
    ]

    (track_id, offset), best = max(best_offsets, key=lambda x: x[1])

    score_gap = best / max(1, second)
    confidence = _confidence(
        best,
        second,
        len(query_hashes),
        len(query_offsets_by_hash),
    )

    accepted = (
        best >= params.min_aligned_matches
        and (best / max(1, len(query_hashes))) >= params.min_query_coverage
        and score_gap >= params.min_score_gap
    )

    return asdict(
        MatchDecision(
            matched=accepted,
            track_id=int(track_id),
            offset=int(offset),
            aligned_matches=int(best),
            second_aligned_matches=int(second),
            query_hashes=len(query_hashes),
            unique_query_hashes=len(query_offsets_by_hash),
            confidence=confidence,
            score_gap=round(float(score_gap), 4),
            reason=None if accepted else "low_confidence",
            top_candidates=[
                {
                    "track_id": int(candidate_track_id),
                    "aligned_matches": int(score),
                    "total_matches": int(track_total_scores[candidate_track_id]),
                }
                for candidate_track_id, score in ranked_tracks[:5]
            ],
        )
    )