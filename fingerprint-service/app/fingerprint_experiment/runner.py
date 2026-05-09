import json
import logging
import random
import time
import zlib
from pathlib import Path, PurePosixPath

import numpy as np

from fingerprint_experiment.algorithm import (
    build_memory_index,
    fingerprint_audio_array,
    match_in_memory,
)
from fingerprint_experiment.augment import apply_corruption
from fingerprint_experiment.audio_ops import (
    cut_random_segment,
    load_s3_audio,
    normalize,
)
from fingerprint_experiment.metrics import build_metrics
from fingerprint_experiment.schemas import FingerprintExperimentRequest
from fingerprint_experiment.utils import write_json, write_jsonl
from repository.s3_client import get_s3

logger = logging.getLogger("fingerprint.experiment")

AUDIO_EXT = (".mp3", ".wav", ".flac", ".ogg")

RUNS_DIR = Path("/app/artifacts/fingerprint-experiments/runs")
FIXED_SETS_DIR = Path("/app/artifacts/fingerprint-experiments/fixed_sets")


def extract_track_id_from_s3_key(s3_key: str) -> int:
    filename = PurePosixPath(s3_key).name
    stem = filename.split(".")[0]

    if stem.isdigit():
        return int(stem)

    return zlib.crc32(s3_key.encode("utf-8")) & 0x7FFFFFFF


def list_s3_audio_keys(bucket: str, prefix: str, limit: int) -> list[str]:
    s3 = get_s3()
    keys = []

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(AUDIO_EXT):
                keys.append(key)

    keys.sort()
    return keys[:limit]


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def make_run_dir(experiment_name: str) -> Path:
    run_id = f"{_safe_name(experiment_name)}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def fixed_eval_set_path(name: str) -> Path:
    FIXED_SETS_DIR.mkdir(parents=True, exist_ok=True)
    return FIXED_SETS_DIR / f"{_safe_name(name)}.json"


def _correct_top5(prediction: dict, gt_track_id: int | None, is_positive: bool) -> bool:
    if not is_positive or gt_track_id is None:
        return False

    ids = [
        int(x["track_id"])
        for x in prediction.get("top_candidates") or []
        if x.get("track_id") is not None
    ]

    return int(gt_track_id) in ids[:5]


def _make_query_spec(
    *,
    query_id: str,
    track_id: int | None,
    duration_sec: float,
    bucket: str,
    corruption: str,
    is_positive: bool,
    seed: int,
) -> dict:
    return {
        "query_id": query_id,
        "track_id": track_id,
        "duration_sec": duration_sec,
        "bucket": bucket,
        "corruption": corruption,
        "is_positive": is_positive,
        "seed": seed,
    }


def build_or_load_fixed_query_specs(
    payload: FingerprintExperimentRequest,
    tracks: dict[int, dict],
) -> tuple[list[dict], Path, bool]:
    path = fixed_eval_set_path(payload.fixed_eval_set_name)
    available_track_ids = set(tracks.keys())

    if path.exists() and not payload.recreate_eval_set:
        with path.open("r", encoding="utf-8") as f:
            specs = json.load(f)

        specs = [
            spec
            for spec in specs
            if spec.get("track_id") is None or int(spec["track_id"]) in available_track_ids
        ]

        if payload.query_limit is not None:
            specs = specs[: payload.query_limit]

        logger.info(
            "Loaded fixed eval set name=%s path=%s queries=%s",
            payload.fixed_eval_set_name,
            path,
            len(specs),
        )

        return specs, path, False

    rng = random.Random(payload.random_seed)
    specs = []

    track_items = list(tracks.values())

    for track in track_items:
        track_id = int(track["track_id"])

        for duration_sec in payload.durations_sec:
            for i in range(payload.clean_queries_per_track):
                specs.append(
                    _make_query_spec(
                        query_id=f"track_{track_id}_clean_{duration_sec}_{i}",
                        track_id=track_id,
                        duration_sec=duration_sec,
                        bucket="clean",
                        corruption="clean",
                        is_positive=True,
                        seed=rng.randint(0, 2**31 - 1),
                    )
                )

            for corruption_idx, corruption in enumerate(payload.noisy_corruptions):
                for i in range(payload.noisy_queries_per_track):
                    specs.append(
                        _make_query_spec(
                            query_id=f"track_{track_id}_{corruption}_{duration_sec}_{i}",
                            track_id=track_id,
                            duration_sec=duration_sec,
                            bucket="dirty",
                            corruption=corruption,
                            is_positive=True,
                            seed=rng.randint(0, 2 ** 31 - 1),
                        )
                    )

    for i in range(payload.negative_queries):
        duration_sec = rng.choice(payload.durations_sec)

        specs.append(
            _make_query_spec(
                query_id=f"negative_white_noise_{i}",
                track_id=None,
                duration_sec=duration_sec,
                bucket="negative",
                corruption="white_noise",
                is_positive=False,
                seed=rng.randint(0, 2**31 - 1),
            )
        )

    rng.shuffle(specs)

    if payload.query_limit is not None:
        specs = specs[: payload.query_limit]

    with path.open("w", encoding="utf-8") as f:
        json.dump(specs, f, ensure_ascii=False, indent=2)

    logger.info(
        "Created fixed eval set name=%s path=%s queries=%s",
        payload.fixed_eval_set_name,
        path,
        len(specs),
    )

    return specs, path, True


def materialize_query_rows(
    specs: list[dict],
    tracks: dict[int, dict],
    payload: FingerprintExperimentRequest,
) -> list[dict]:
    params = payload.params
    rows = []

    for spec in specs:
        seed = int(spec["seed"])
        random.seed(seed)
        np.random.seed(seed)

        duration_sec = float(spec["duration_sec"])
        corruption = str(spec["corruption"])
        is_positive = bool(spec["is_positive"])

        if is_positive:
            track_id = int(spec["track_id"])
            audio = tracks[track_id]["audio"]

            segment = cut_random_segment(
                audio=audio,
                sample_rate=params.sample_rate,
                duration_sec=duration_sec,
            )

            query_audio = apply_corruption(
                segment,
                corruption,
                params.sample_rate,
            )

        else:
            samples = int(duration_sec * params.sample_rate)
            query_audio = normalize(np.random.randn(samples).astype(np.float32))

        rows.append(
            {
                "query_id": spec["query_id"],
                "track_id": spec.get("track_id"),
                "duration_sec": duration_sec,
                "bucket": spec["bucket"],
                "corruption": corruption,
                "is_positive": is_positive,
                "seed": seed,
                "audio": query_audio,
            }
        )

    return rows


def run_fingerprint_experiment(payload: FingerprintExperimentRequest) -> dict:
    started = time.time()

    random.seed(payload.random_seed)
    np.random.seed(payload.random_seed)

    run_dir = make_run_dir(payload.experiment_name)
    params = payload.params

    write_json(run_dir / "params.json", payload.model_dump())

    logger.info(
        "Fingerprint experiment started run_dir=%s bucket=%s prefix=%s track_limit=%s query_limit=%s fixed_eval_set=%s recreate_eval_set=%s params=%s",
        run_dir,
        payload.bucket,
        payload.prefix,
        payload.track_limit,
        payload.query_limit,
        payload.fixed_eval_set_name,
        payload.recreate_eval_set,
        params.model_dump(),
    )

    keys = list_s3_audio_keys(
        bucket=payload.bucket,
        prefix=payload.prefix,
        limit=payload.track_limit,
    )

    if len(keys) < 2:
        raise ValueError(f"Need at least 2 audio tracks, got {len(keys)}")

    logger.info("S3 keys loaded count=%s", len(keys))

    tracks: dict[int, dict] = {}
    track_hashes: dict[int, list[tuple[int, int]]] = {}

    index_started = time.perf_counter()

    for idx, key in enumerate(keys, start=1):
        track_id = extract_track_id_from_s3_key(key)

        audio, duration_sec = load_s3_audio(
            bucket=payload.bucket,
            key=key,
            sample_rate=params.sample_rate,
        )

        index_audio = audio
        if payload.index_duration_sec is not None:
            max_len = int(payload.index_duration_sec * params.sample_rate)
            index_audio = audio[:max_len]

        fp = fingerprint_audio_array(index_audio, params)

        tracks[track_id] = {
            "track_id": track_id,
            "s3_key": key,
            "duration_sec": duration_sec,
            "samples": int(len(audio)),
            "hashes": len(fp["hashes"]),
            "peaks": fp["peaks"],
            "frames": fp["frames"],
            "audio": audio,
        }

        track_hashes[track_id] = fp["hashes"]

        if idx % 25 == 0 or idx == len(keys):
            logger.info(
                "Index build progress tracks=%s/%s avg_hashes=%.1f avg_peaks=%.1f elapsed_sec=%.1f",
                idx,
                len(keys),
                float(np.mean([x["hashes"] for x in tracks.values()])),
                float(np.mean([x["peaks"] for x in tracks.values()])),
                time.perf_counter() - index_started,
            )

    index = build_memory_index(track_hashes)

    logger.info(
        "In-memory index built tracks=%s unique_hashes=%s elapsed_sec=%.2f",
        len(tracks),
        len(index),
        time.perf_counter() - index_started,
    )

    query_specs, fixed_set_path, fixed_set_created = build_or_load_fixed_query_specs(
        payload=payload,
        tracks=tracks,
    )

    query_rows = materialize_query_rows(
        specs=query_specs,
        tracks=tracks,
        payload=payload,
    )

    logger.info(
        "Queries materialized count=%s fixed_eval_set_path=%s fixed_eval_set_created=%s",
        len(query_rows),
        fixed_set_path,
        fixed_set_created,
    )

    result_rows = []

    for idx, query in enumerate(query_rows, start=1):
        query_started = time.perf_counter()

        fp = fingerprint_audio_array(query["audio"], params)
        prediction = match_in_memory(fp["hashes"], index, params)

        latency_ms = (time.perf_counter() - query_started) * 1000.0

        gt = query["track_id"]
        is_positive = bool(query["is_positive"])
        predicted = prediction.get("track_id")

        correct_top1 = bool(
            is_positive
            and prediction.get("matched")
            and predicted is not None
            and gt is not None
            and int(predicted) == int(gt)
        )

        correct_top5 = _correct_top5(prediction, gt, is_positive)

        result_rows.append(
            {
                "query_id": query["query_id"],
                "bucket": query["bucket"],
                "corruption": query["corruption"],
                "duration_sec": query["duration_sec"],
                "is_positive": is_positive,
                "ground_truth_track_id": gt,
                "predicted_track_id": predicted,
                "matched": bool(prediction.get("matched")),
                "correct_top1": correct_top1,
                "correct_top5": correct_top5,
                "confidence": prediction.get("confidence"),
                "aligned_matches": prediction.get("aligned_matches"),
                "second_aligned_matches": prediction.get("second_aligned_matches"),
                "query_hashes": prediction.get("query_hashes"),
                "unique_query_hashes": prediction.get("unique_query_hashes"),
                "score_gap": prediction.get("score_gap"),
                "reason": prediction.get("reason"),
                "latency_ms": latency_ms,
                "top_candidates": prediction.get("top_candidates") or [],
                "raw_prediction": prediction,
                "seed": query.get("seed"),
            }
        )

        if idx % 100 == 0 or idx == len(query_rows):
            logger.info(
                "Experiment eval progress queries=%s/%s elapsed_sec=%.1f",
                idx,
                len(query_rows),
                time.time() - started,
            )

    metrics = build_metrics(result_rows)

    track_summaries = [
        {k: v for k, v in track.items() if k != "audio"}
        for track in tracks.values()
    ]

    index_summary = {
        "tracks": len(tracks),
        "unique_hashes": len(index),
        "avg_hashes_per_track": float(np.mean([x["hashes"] for x in tracks.values()])),
        "avg_peaks_per_track": float(np.mean([x["peaks"] for x in tracks.values()])),
        "track_summaries": track_summaries,
    }

    summary = {
        "run_dir": str(run_dir),
        "params_path": str(run_dir / "params.json"),
        "results_path": str(run_dir / "results.jsonl"),
        "metrics_path": str(run_dir / "metrics.json"),
        "index_summary_path": str(run_dir / "index_summary.json"),
        "fixed_eval_set_path": str(fixed_set_path),
        "fixed_eval_set_created": fixed_set_created,
        "fixed_eval_set_name": payload.fixed_eval_set_name,
        "tracks": len(tracks),
        "queries": len(result_rows),
        "elapsed_sec": time.time() - started,
        "metrics": metrics,
        "index_summary": {
            "tracks": index_summary["tracks"],
            "unique_hashes": index_summary["unique_hashes"],
            "avg_hashes_per_track": index_summary["avg_hashes_per_track"],
            "avg_peaks_per_track": index_summary["avg_peaks_per_track"],
        },
    }

    write_jsonl(run_dir / "results.jsonl", result_rows)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "index_summary.json", index_summary)
    write_json(run_dir / "summary.json", summary)

    logger.info("Fingerprint experiment finished summary=%s", summary)

    return summary