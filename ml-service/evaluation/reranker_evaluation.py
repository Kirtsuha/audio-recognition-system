from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import boto3
import numpy as np
import requests
import soundfile as sf

from pipeline.augment import augment_audio_strong_noisy, augment_audio_phone_noisy
from pipeline.config import EVAL_DIR, SR
from pipeline.dataset import pad_or_trim, random_segment
from pipeline.manifest import load_prepared_manifest

logger = logging.getLogger("ml-pipeline.evaluate_reranker")


FINGERPRINT_SERVICE_URL = os.getenv("FINGERPRINT_SERVICE_URL", "http://fingerprint-service:8000")
ML_RERANKER_SERVICE_URL = os.getenv("ML_RERANKER_SERVICE_URL", "http://ml-reranker-service:8010")

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio123")

RERANK_EVAL_QUERY_BUCKET = os.getenv("RERANK_EVAL_QUERY_BUCKET", "recognition-queries")
RERANK_EVAL_REFERENCE_BUCKET = os.getenv("RERANK_EVAL_REFERENCE_BUCKET", "full-tracks")


class S3Uploader:
    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self.client.head_bucket(Bucket=bucket)
        except Exception:
            self.client.create_bucket(Bucket=bucket)

    def put_wav(self, bucket: str, key: str, audio: np.ndarray, sr: int) -> None:
        self.ensure_bucket(bucket)

        bio = io.BytesIO()
        sf.write(bio, audio.astype("float32"), sr, format="WAV")
        data = bio.getvalue()

        self.client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType="audio/wav",
        )


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def make_query_audio(clean_query: np.ndarray, case: str) -> np.ndarray:
    case = case.lower().strip()

    if case == "clean":
        return clean_query.astype("float32")

    if case == "noisy":
        return augment_audio_strong_noisy(clean_query).astype("float32")

    if case == "phone_noisy":
        return augment_audio_phone_noisy(clean_query).astype("float32")

    raise ValueError(f"Unsupported case={case}")


def call_fingerprint_retrieve_candidates(
    *,
    audio: np.ndarray,
    sr: int,
    top_k: int,
    timeout_sec: float,
) -> dict[str, Any]:
    bio = io.BytesIO()
    sf.write(bio, audio.astype("float32"), sr, format="WAV")
    bio.seek(0)

    response = requests.post(
        f"{FINGERPRINT_SERVICE_URL}/retrieve-candidates",
        params={"top_k": top_k},
        files={"file": ("query.wav", bio.getvalue(), "audio/wav")},
        timeout=timeout_sec,
    )
    response.raise_for_status()
    return response.json()


def call_ml_reranker(
    *,
    request_id: str,
    query_bucket: str,
    query_key: str,
    reference_bucket: str,
    fingerprint_response: dict[str, Any],
    max_candidates: int,
    segment_seconds: float,
    fp_weight: float,
    ml_weight: float,
    timeout_sec: float,
) -> dict[str, Any]:
    payload = {
        "request_id": request_id,
        "query_audio": {
            "bucket": query_bucket,
            "key": query_key,
        },
        "reference_bucket": reference_bucket,
        "fingerprint": fingerprint_response,
        "options": {
            "max_candidates": max_candidates,
            "segment_seconds": segment_seconds,
            # Если fingerprint уже отдает best_offset_sec, это поле не используется.
            "fingerprint_offset_hop_seconds": 1.0,
            "offset_jitter_seconds": [0.0],
            "fp_weight": fp_weight,
            "ml_weight": ml_weight,
            "no_match_threshold": 0.0,
        },
    }

    response = requests.post(
        f"{ML_RERANKER_SERVICE_URL}/rerank",
        json=payload,
        timeout=timeout_sec,
    )
    response.raise_for_status()
    return response.json()


def candidate_track_ids(candidates: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for item in candidates:
        try:
            ids.append(int(item["track_id"]))
        except Exception:
            continue
    return ids


def evaluate_one_case(
    *,
    items: list[dict[str, Any]],
    case_name: str,
    query_limit: int,
    top_k: int,
    reranker_max_candidates: int,
    segment_seconds: float,
    fp_weight: float,
    ml_weight: float,
    output_jsonl_path: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    uploader = S3Uploader()

    rows: list[dict[str, Any]] = []
    fingerprint_latencies: list[float] = []
    reranker_latencies: list[float] = []
    total_latencies: list[float] = []

    total = 0

    fingerprint_top1_correct = 0
    fingerprint_top5_correct = 0
    fingerprint_topk_contains_true = 0

    reranker_top1_correct = 0
    reranker_top5_correct = 0
    reranker_topk_contains_true = 0

    improved = 0
    worsened = 0
    unchanged = 0
    false_no_match = 0

    rerankable_count = 0
    rerankable_reranker_top1_correct = 0

    selected = items[:query_limit]

    for idx, row in enumerate(selected, start=1):
        started_total = time.perf_counter()
        track_id = int(row["track_id"])
        audio = np.load(row["prepared_path"]).astype("float32")

        clean_query = pad_or_trim(random_segment(audio), int(round(segment_seconds * SR)))
        query_audio = make_query_audio(clean_query, case_name)
        query_audio = pad_or_trim(query_audio, int(round(segment_seconds * SR)))

        request_id = f"rerank-eval-{case_name}-{uuid4().hex}"
        query_key = f"rerank-eval/{case_name}/{request_id}.wav"

        uploader.put_wav(
            bucket=RERANK_EVAL_QUERY_BUCKET,
            key=query_key,
            audio=query_audio,
            sr=SR,
        )

        fp_started = time.perf_counter()
        fp_response = call_fingerprint_retrieve_candidates(
            audio=query_audio,
            sr=SR,
            top_k=top_k,
            timeout_sec=timeout_sec,
        )
        fp_latency_ms = (time.perf_counter() - fp_started) * 1000.0
        fingerprint_latencies.append(fp_latency_ms)

        fp_candidates = fp_response.get("candidates") or []
        fp_ranked = candidate_track_ids(fp_candidates)

        fp_top1 = fp_ranked[0] if fp_ranked else None
        fp_correct_top1 = bool(fp_top1 == track_id)
        fp_correct_top5 = bool(track_id in fp_ranked[:5])
        fp_contains_true = bool(track_id in fp_ranked[:top_k])

        fingerprint_top1_correct += int(fp_correct_top1)
        fingerprint_top5_correct += int(fp_correct_top5)
        fingerprint_topk_contains_true += int(fp_contains_true)

        rr_started = time.perf_counter()
        rr_response = call_ml_reranker(
            request_id=request_id,
            query_bucket=RERANK_EVAL_QUERY_BUCKET,
            query_key=query_key,
            reference_bucket=RERANK_EVAL_REFERENCE_BUCKET,
            fingerprint_response=fp_response,
            max_candidates=reranker_max_candidates,
            segment_seconds=segment_seconds,
            fp_weight=fp_weight,
            ml_weight=ml_weight,
            timeout_sec=timeout_sec,
        )
        rr_latency_ms = (time.perf_counter() - rr_started) * 1000.0
        reranker_latencies.append(rr_latency_ms)

        rr_candidates = rr_response.get("candidates") or []
        rr_ranked = candidate_track_ids(rr_candidates)
        rr_top1 = rr_ranked[0] if rr_ranked else None

        rr_correct_top1 = bool(rr_top1 == track_id)
        rr_correct_top5 = bool(track_id in rr_ranked[:5])
        rr_contains_true = bool(track_id in rr_ranked[:reranker_max_candidates])

        reranker_top1_correct += int(rr_correct_top1)
        reranker_top5_correct += int(rr_correct_top5)
        reranker_topk_contains_true += int(rr_contains_true)

        if fp_contains_true:
            rerankable_count += 1
            rerankable_reranker_top1_correct += int(rr_correct_top1)

        if not fp_correct_top1 and rr_correct_top1:
            improved += 1
        elif fp_correct_top1 and not rr_correct_top1:
            worsened += 1
        else:
            unchanged += 1

        if not rr_response.get("matched", False) and fp_contains_true:
            false_no_match += 1

        total_latency_ms = (time.perf_counter() - started_total) * 1000.0
        total_latencies.append(total_latency_ms)

        result_row = {
            "request_id": request_id,
            "case": case_name,
            "track_id": track_id,
            "s3_key": row.get("s3_key"),
            "fingerprint_top1": fp_top1,
            "fingerprint_ranked": fp_ranked[:top_k],
            "fingerprint_top1_correct": fp_correct_top1,
            "fingerprint_top5_correct": fp_correct_top5,
            "fingerprint_topk_contains_true": fp_contains_true,
            "reranker_top1": rr_top1,
            "reranker_ranked": rr_ranked[:reranker_max_candidates],
            "reranker_top1_correct": rr_correct_top1,
            "reranker_top5_correct": rr_correct_top5,
            "reranker_topk_contains_true": rr_contains_true,
            "reranker_matched": bool(rr_response.get("matched", False)),
            "reranker_best": rr_response.get("best"),
            "fp_latency_ms": fp_latency_ms,
            "reranker_latency_ms": rr_latency_ms,
            "total_latency_ms": total_latency_ms,
        }
        rows.append(result_row)

        total += 1

        if len(rows) >= 25:
            append_jsonl(output_jsonl_path, rows)
            rows = []

        if idx % 25 == 0 or idx == len(selected):
            logger.info(
                "Reranker eval progress case=%s %s/%s fp_top1=%.4f rr_top1=%.4f improved=%s worsened=%s",
                case_name,
                idx,
                len(selected),
                fingerprint_top1_correct / max(total, 1),
                reranker_top1_correct / max(total, 1),
                improved,
                worsened,
            )

    if rows:
        append_jsonl(output_jsonl_path, rows)

    return {
        "case": case_name,
        "count": total,
        "fingerprint_top1_accuracy": fingerprint_top1_correct / total if total else 0.0,
        "fingerprint_top5_accuracy": fingerprint_top5_correct / total if total else 0.0,
        "fingerprint_topk_recall": fingerprint_topk_contains_true / total if total else 0.0,
        "reranker_top1_accuracy": reranker_top1_correct / total if total else 0.0,
        "reranker_top5_accuracy": reranker_top5_correct / total if total else 0.0,
        "reranker_topk_recall": reranker_topk_contains_true / total if total else 0.0,
        "rerankable_count": rerankable_count,
        "rerankable_ratio": rerankable_count / total if total else 0.0,
        "rerankable_reranker_top1_accuracy": (
            rerankable_reranker_top1_correct / rerankable_count if rerankable_count else 0.0
        ),
        "reranker_improved_count": improved,
        "reranker_worsened_count": worsened,
        "reranker_unchanged_count": unchanged,
        "false_no_match_count": false_no_match,
        "fingerprint_latency_p50_ms": percentile(fingerprint_latencies, 50),
        "fingerprint_latency_p95_ms": percentile(fingerprint_latencies, 95),
        "reranker_latency_p50_ms": percentile(reranker_latencies, 50),
        "reranker_latency_p95_ms": percentile(reranker_latencies, 95),
        "total_latency_p50_ms": percentile(total_latencies, 50),
        "total_latency_p95_ms": percentile(total_latencies, 95),
    }


def evaluate_reranker(
    *,
    output_metrics_path: Path,
    output_results_path: Path,
    val_limit: int,
    test_limit: int,
    query_limit: int,
    top_k: int,
    reranker_max_candidates: int,
    cases: list[str],
    segment_seconds: float,
    fp_weight: float,
    ml_weight: float,
    timeout_sec: float,
) -> dict[str, Any]:
    started = time.time()

    items = load_prepared_manifest(
        train_limit=0,
        val_limit=val_limit,
        test_limit=test_limit,
    )

    # Prefer val split for comparable fixed validation.
    val_items = [x for x in items if x.get("split") == "val"]
    if not val_items:
        val_items = items

    val_items = val_items[:query_limit]

    if len(val_items) < 1:
        raise ValueError("No eval items found")

    if output_results_path.exists():
        output_results_path.unlink()

    logger.info(
        "Reranker evaluation started items=%s cases=%s fp_url=%s rr_url=%s top_k=%s rr_max=%s weights=%s/%s",
        len(val_items),
        cases,
        FINGERPRINT_SERVICE_URL,
        ML_RERANKER_SERVICE_URL,
        top_k,
        reranker_max_candidates,
        fp_weight,
        ml_weight,
    )

    case_metrics: dict[str, Any] = {}

    for case_name in cases:
        case_metrics[case_name] = evaluate_one_case(
            items=val_items,
            case_name=case_name,
            query_limit=query_limit,
            top_k=top_k,
            reranker_max_candidates=reranker_max_candidates,
            segment_seconds=segment_seconds,
            fp_weight=fp_weight,
            ml_weight=ml_weight,
            output_jsonl_path=output_results_path,
            timeout_sec=timeout_sec,
        )

    metrics = {
        "count": len(val_items),
        "cases": case_metrics,
        "output_results_path": str(output_results_path),
        "val_limit": val_limit,
        "test_limit": test_limit,
        "query_limit": query_limit,
        "top_k": top_k,
        "reranker_max_candidates": reranker_max_candidates,
        "segment_seconds": segment_seconds,
        "fp_weight": fp_weight,
        "ml_weight": ml_weight,
        "fingerprint_service_url": FINGERPRINT_SERVICE_URL,
        "ml_reranker_service_url": ML_RERANKER_SERVICE_URL,
        "query_bucket": RERANK_EVAL_QUERY_BUCKET,
        "reference_bucket": RERANK_EVAL_REFERENCE_BUCKET,
        "elapsed_sec": time.time() - started,
    }

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Reranker evaluation finished metrics_path=%s metrics=%s", output_metrics_path, metrics)
    return metrics


def parse_cases(raw: str) -> list[str]:
    out = [x.strip() for x in raw.split(",") if x.strip()]
    return out or ["clean"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-metrics", default=str(EVAL_DIR / "reranker_metrics.json"))
    parser.add_argument("--output-results", default=str(EVAL_DIR / "reranker_results.jsonl"))
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--reranker-max-candidates", type=int, default=20)
    parser.add_argument("--cases", default="clean,noisy,phone_noisy")
    parser.add_argument("--segment-seconds", type=float, default=15.0)
    parser.add_argument("--fp-weight", type=float, default=0.95)
    parser.add_argument("--ml-weight", type=float, default=0.05)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    evaluate_reranker(
        output_metrics_path=Path(args.output_metrics),
        output_results_path=Path(args.output_results),
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        query_limit=args.query_limit,
        top_k=args.top_k,
        reranker_max_candidates=args.reranker_max_candidates,
        cases=parse_cases(args.cases),
        segment_seconds=args.segment_seconds,
        fp_weight=args.fp_weight,
        ml_weight=args.ml_weight,
        timeout_sec=args.timeout_sec,
    )


if __name__ == "__main__":
    main()