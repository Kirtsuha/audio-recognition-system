import logging
import random
import time

import numpy as np
import soundfile as sf

from app.audio import (
    apply_corruption,
    cut_random_segment,
    load_prepared_audio,
    load_s3_audio,
    normalize_peak,
)
from app.config import (
    DATASET_PROGRESS_EVERY,
    DEFAULT_CORRUPTIONS,
    EVAL_AUDIO_SOURCE,
    EVAL_QUERIES_DIR,
    EVAL_QUERIES_PATH,
    PREPARED_MANIFEST_PATH,
    SR,
)
from app.logging_utils import log_stage
from app.utils import read_jsonl, write_jsonl

logger = logging.getLogger("evaluation.dataset")


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


def _load_eval_audio(item: dict) -> tuple[np.ndarray, str]:
    if EVAL_AUDIO_SOURCE == "s3":
        s3_key = item.get("s3_key")
        if not s3_key:
            raise ValueError(f"Manifest item has no s3_key: {item}")
        return load_s3_audio(s3_key), "s3"

    if EVAL_AUDIO_SOURCE == "prepared":
        return load_prepared_audio(item["prepared_path"]), "prepared"

    raise ValueError(f"Unsupported EVAL_AUDIO_SOURCE: {EVAL_AUDIO_SOURCE}")


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
    corruptions: list[str] | None = None,
    seed: int | None = 42,
) -> dict:
    started = time.perf_counter()
    corruptions = corruptions or DEFAULT_CORRUPTIONS

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    with log_stage(
        logger,
        "generate-eval-dataset",
        audio_source=EVAL_AUDIO_SOURCE,
        test_limit=test_limit,
        negative_limit=negative_limit,
        clean_per_track=per_track_clean_queries,
        noisy_per_track=per_track_noisy_queries,
        durations=durations_sec,
        corruptions=corruptions,
        seed=seed,
    ):
        items = _load_test_items(test_limit)

        logger.info(
            "Loaded test items count=%s manifest=%s",
            len(items),
            PREPARED_MANIFEST_PATH,
        )

        rows = []
        failed_items = 0
        positive_queries = 0

        for item_idx, item in enumerate(items, start=1):
            track_id = int(item["track_id"])
            s3_key = item.get("s3_key")

            try:
                audio, audio_source = _load_eval_audio(item)
            except Exception as exc:
                failed_items += 1
                logger.warning(
                    "Failed to load eval audio track_id=%s s3_key=%s error=%s",
                    track_id,
                    s3_key,
                    exc,
                )
                continue

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
                            "s3_key": s3_key,
                            "bucket": "clean",
                            "corruption": "clean",
                            "duration_sec": duration_sec,
                            "is_positive": True,
                            "audio_source": audio_source,
                        }
                    )
                    positive_queries += 1

                noisy_corruptions = [x for x in corruptions if x != "clean"]

                for noisy_idx in range(per_track_noisy_queries):
                    if not noisy_corruptions:
                        break

                    corruption = noisy_corruptions[noisy_idx % len(noisy_corruptions)]

                    segment = cut_random_segment(audio, duration_sec)
                    query_audio = apply_corruption(segment, corruption)

                    query_id = f"track_{track_id}_{corruption}_{duration_sec:.1f}s_{noisy_idx}"
                    query_path = _save_query_audio(query_audio, query_id)

                    rows.append(
                        {
                            "query_id": query_id,
                            "query_path": query_path,
                            "track_id": track_id,
                            "s3_key": s3_key,
                            "bucket": "dirty",
                            "corruption": corruption,
                            "duration_sec": duration_sec,
                            "is_positive": True,
                            "audio_source": audio_source,
                        }
                    )
                    positive_queries += 1

            if item_idx % DATASET_PROGRESS_EVERY == 0 or item_idx == len(items):
                logger.info(
                    "Dataset generation progress tracks=%s/%s rows=%s positive=%s failed_items=%s",
                    item_idx,
                    len(items),
                    len(rows),
                    positive_queries,
                    failed_items,
                )

        negative_items = items[:negative_limit]
        negative_queries = 0

        logger.info("Generating negative queries items=%s", len(negative_items))

        for item_idx, item in enumerate(negative_items, start=1):
            track_id = int(item["track_id"])
            s3_key = item.get("s3_key")

            try:
                audio, audio_source = _load_eval_audio(item)
            except Exception as exc:
                logger.warning(
                    "Failed to load negative source track_id=%s s3_key=%s error=%s",
                    track_id,
                    s3_key,
                    exc,
                )
                continue

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
                        "audio_source": audio_source,
                    }
                )
                negative_queries += 1

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
                        "audio_source": audio_source,
                    }
                )
                negative_queries += 1

            if item_idx % DATASET_PROGRESS_EVERY == 0 or item_idx == len(negative_items):
                logger.info(
                    "Negative generation progress tracks=%s/%s negative_queries=%s total_rows=%s",
                    item_idx,
                    len(negative_items),
                    negative_queries,
                    len(rows),
                )

        random.shuffle(rows)
        write_jsonl(EVAL_QUERIES_PATH, rows)

        elapsed = time.perf_counter() - started

        summary = {
            "queries": len(rows),
            "positive_queries": sum(1 for x in rows if x["is_positive"]),
            "negative_queries": sum(1 for x in rows if not x["is_positive"]),
            "failed_items": failed_items,
            "queries_path": str(EVAL_QUERIES_PATH),
            "queries_dir": str(EVAL_QUERIES_DIR),
            "audio_source": EVAL_AUDIO_SOURCE,
            "corruptions": corruptions,
            "seed": seed,
            "elapsed_sec": elapsed,
        }

        logger.info("Eval dataset generated summary=%s", summary)

        return summary
