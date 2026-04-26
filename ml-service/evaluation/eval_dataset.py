import logging
import random
import tempfile

import numpy as np
import soundfile as sf

from pipeline.augment import (
    add_gaussian_noise,
    normalize_peak,
    random_clipping,
    random_gain,
    random_pitch_shift,
    random_time_stretch,
    simple_reverb,
)
from pipeline.config import EVAL_QUERIES_PATH, SR
from pipeline.manifest import iter_prepared_manifest
from evaluation.eval_utils import write_jsonl

logger = logging.getLogger("ml-pipeline.eval-dataset")


def _cut_random_segment(audio: np.ndarray, duration_sec: float) -> np.ndarray:
    target_len = int(duration_sec * SR)

    if len(audio) <= target_len:
        out = np.zeros(target_len, dtype=np.float32)
        out[: len(audio)] = audio.astype(np.float32)
        return out

    max_start = len(audio) - target_len
    start = random.randint(0, max_start)
    end = start + target_len
    return audio[start:end].astype(np.float32)


def _apply_corruption(audio: np.ndarray, corruption: str) -> np.ndarray:
    out = audio.astype(np.float32).copy()

    if corruption == "clean":
        return normalize_peak(out)

    if corruption == "noise_snr20":
        out = add_gaussian_noise(out, snr_db_min=18.0, snr_db_max=22.0)
    elif corruption == "noise_snr10":
        out = add_gaussian_noise(out, snr_db_min=9.0, snr_db_max=11.0)
    elif corruption == "noise_snr5":
        out = add_gaussian_noise(out, snr_db_min=4.0, snr_db_max=6.0)
    elif corruption == "reverb":
        out = simple_reverb(out)
    elif corruption == "clipping":
        out = random_clipping(out, min_clip=0.55, max_clip=0.75)
    elif corruption == "gain":
        out = random_gain(out, min_db=-12.0, max_db=8.0)
    elif corruption == "pitch_shift":
        out = random_pitch_shift(out, sr=SR)
    elif corruption == "time_stretch":
        out = random_time_stretch(out)
    elif corruption == "mixed_noise_reverb":
        out = add_gaussian_noise(out, snr_db_min=8.0, snr_db_max=14.0)
        out = simple_reverb(out)
    else:
        raise ValueError(f"Unsupported corruption: {corruption}")

    return normalize_peak(out)


def _save_query_audio(audio: np.ndarray, query_id: str) -> str:
    tmp = tempfile.NamedTemporaryFile(prefix=f"{query_id}_", suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    sf.write(tmp_path, audio.astype(np.float32), SR)
    return tmp_path


def generate_eval_dataset(
    per_track_clean_queries: int,
    per_track_noisy_queries: int,
    durations_sec: list[float],
) -> dict:
    items = [row for row in iter_prepared_manifest() if row["split"] == "test"]
    if not items:
        raise ValueError("No test items found in prepared manifest")

    logger.info(
        "Generate eval dataset started test_items=%s clean_per_track=%s noisy_per_track=%s durations=%s",
        len(items),
        per_track_clean_queries,
        per_track_noisy_queries,
        durations_sec,
    )

    rows: list[dict] = []

    noisy_buckets = [
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

    for item_idx, row in enumerate(items, start=1):
        track_id = int(row["track_id"])
        prepared_path = row["prepared_path"]

        audio = np.load(prepared_path).astype(np.float32)

        for duration_sec in durations_sec:
            for clean_idx in range(per_track_clean_queries):
                seg = _cut_random_segment(audio, duration_sec)
                clean_audio = _apply_corruption(seg, "clean")
                query_id = f"track_{track_id}_clean_{duration_sec:.1f}s_{clean_idx}"
                query_path = _save_query_audio(clean_audio, query_id)

                rows.append(
                    {
                        "query_id": query_id,
                        "query_path": query_path,
                        "track_id": track_id,
                        "s3_key": row["s3_key"],
                        "bucket": "clean",
                        "corruption": "clean",
                        "duration_sec": duration_sec,
                        "is_positive": True,
                    }
                )

            for noisy_idx in range(per_track_noisy_queries):
                corruption = noisy_buckets[noisy_idx % len(noisy_buckets)]
                seg = _cut_random_segment(audio, duration_sec)
                noisy_audio = _apply_corruption(seg, corruption)
                query_id = f"track_{track_id}_{corruption}_{duration_sec:.1f}s_{noisy_idx}"
                query_path = _save_query_audio(noisy_audio, query_id)

                rows.append(
                    {
                        "query_id": query_id,
                        "query_path": query_path,
                        "track_id": track_id,
                        "s3_key": row["s3_key"],
                        "bucket": "dirty",
                        "corruption": corruption,
                        "duration_sec": duration_sec,
                        "is_positive": True,
                    }
                )

        if item_idx % 100 == 0 or item_idx == len(items):
            logger.info("Generate eval dataset progress tracks=%s/%s queries=%s", item_idx, len(items), len(rows))

    # Negative set: take test tracks and destroy semantics strongly
    negative_rows = []
    for item_idx, row in enumerate(items, start=1):
        track_id = int(row["track_id"])
        prepared_path = row["prepared_path"]
        audio = np.load(prepared_path).astype(np.float32)

        for duration_sec in durations_sec:
            seg = _cut_random_segment(audio, duration_sec)

            noise_only = np.random.randn(len(seg)).astype(np.float32)
            noise_only = normalize_peak(noise_only)
            query_id = f"neg_noise_{track_id}_{duration_sec:.1f}s"
            query_path = _save_query_audio(noise_only, query_id)

            negative_rows.append(
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

            reversed_seg = seg[::-1].copy()
            reversed_seg = normalize_peak(reversed_seg)
            query_id = f"neg_reverse_{track_id}_{duration_sec:.1f}s"
            query_path = _save_query_audio(reversed_seg, query_id)

            negative_rows.append(
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

        if item_idx >= min(200, len(items)):
            break

    rows.extend(negative_rows)
    write_jsonl(EVAL_QUERIES_PATH, rows)

    summary = {
        "queries": len(rows),
        "positive_queries": sum(1 for x in rows if x["is_positive"]),
        "negative_queries": sum(1 for x in rows if not x["is_positive"]),
        "queries_path": str(EVAL_QUERIES_PATH),
    }

    logger.info("Generate eval dataset finished summary=%s", summary)
    return summary