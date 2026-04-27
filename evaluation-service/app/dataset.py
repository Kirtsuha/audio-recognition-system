import random
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio import apply_corruption, cut_random_segment, load_prepared_audio, normalize_peak
from app.config import EVAL_QUERIES_DIR, EVAL_QUERIES_PATH, PREPARED_MANIFEST_PATH, SR
from app.utils import read_jsonl, write_jsonl


NOISY_BUCKETS = [
    "noise_snr20",
    "noise_snr10",
    "noise_snr5",
    "reverb",
    "clipping",
    "gain",
    "pitch_shift",
    "time_stretch",
    "mixed_noise_reverb",
]


def _safe_filename(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _save_query_audio(audio: np.ndarray, query_id: str) -> str:
    EVAL_QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_QUERIES_DIR / f"{_safe_filename(query_id)}.wav"
    sf.write(str(path), audio.astype(np.float32), SR)
    return str(path)


def _load_test_items(test_limit: int | None = None) -> list[dict]:
    rows = read_jsonl(PREPARED_MANIFEST_PATH)
    rows = [x for x in rows if x.get("split") == "test"]

    if test_limit is not None:
        rows = rows[:test_limit]

    if not rows:
        raise ValueError(f"No test rows found in manifest: {PREPARED_MANIFEST_PATH}")

    return rows


def generate_eval_dataset(
    per_track_clean_queries: int,
    per_track_noisy_queries: int,
    durations_sec: list[float],
    test_limit: int | None = None,
    negative_limit: int = 200,
) -> dict:
    items = _load_test_items(test_limit)

    rows = []

    for item_idx, item in enumerate(items, start=1):
        track_id = int(item["track_id"])
        audio = load_prepared_audio(item["prepared_path"])

        for duration_sec in durations_sec:
            for clean_idx in range(per_track_clean_queries):
                segment = cut_random_segment(audio, duration_sec)
                query_audio = apply_corruption(segment, "clean")

                query_id = f"track_{track_id}_clean_{duration_sec:.1f}s_{clean_idx}"
                query_path = _save_query_audio(query_audio, query_id)

                rows.append(
                    {
                        "query_id": query_id,
                        "query_path": query_path,
                        "track_id": track_id,
                        "s3_key": item.get("s3_key"),
                        "bucket": "clean",
                        "corruption": "clean",
                        "duration_sec": duration_sec,
                        "is_positive": True,
                    }
                )

            for noisy_idx in range(per_track_noisy_queries):
                corruption = NOISY_BUCKETS[noisy_idx % len(NOISY_BUCKETS)]

                segment = cut_random_segment(audio, duration_sec)
                query_audio = apply_corruption(segment, corruption)

                query_id = f"track_{track_id}_{corruption}_{duration_sec:.1f}s_{noisy_idx}"
                query_path = _save_query_audio(query_audio, query_id)

                rows.append(
                    {
                        "query_id": query_id,
                        "query_path": query_path,
                        "track_id": track_id,
                        "s3_key": item.get("s3_key"),
                        "bucket": "dirty",
                        "corruption": corruption,
                        "duration_sec": duration_sec,
                        "is_positive": True,
                    }
                )

    negative_items = items[:negative_limit]

    for item in negative_items:
        track_id = int(item["track_id"])
        audio = load_prepared_audio(item["prepared_path"])

        for duration_sec in durations_sec:
            segment = cut_random_segment(audio, duration_sec)

            noise = np.random.randn(len(segment)).astype(np.float32)
            noise = normalize_peak(noise)

            query_id = f"neg_noise_{track_id}_{duration_sec:.1f}s"
            query_path = _save_query_audio(noise, query_id)

            rows.append(
                {
                    "query_id": query_id,
                    "query_path": query_path,
                    "track_id": None,
                    "s3_key": None,
                    "bucket": "negative",
                    "corruption": "white_noise",
                    "duration_sec": duration_sec,
                    "is_positive": False,
                }
            )

            reversed_audio = normalize_peak(segment[::-1].copy())
            query_id = f"neg_reverse_{track_id}_{duration_sec:.1f}s"
            query_path = _save_query_audio(reversed_audio, query_id)

            rows.append(
                {
                    "query_id": query_id,
                    "query_path": query_path,
                    "track_id": None,
                    "s3_key": None,
                    "bucket": "negative",
                    "corruption": "reversed_audio",
                    "duration_sec": duration_sec,
                    "is_positive": False,
                }
            )

    random.shuffle(rows)
    write_jsonl(EVAL_QUERIES_PATH, rows)

    return {
        "queries": len(rows),
        "positive_queries": sum(1 for x in rows if x["is_positive"]),
        "negative_queries": sum(1 for x in rows if not x["is_positive"]),
        "queries_path": str(EVAL_QUERIES_PATH),
        "queries_dir": str(EVAL_QUERIES_DIR),
    }