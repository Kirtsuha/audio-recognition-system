import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch

from app.inference import aggregate_results
from app.model import AudioEncoder
from pipeline.augment import (
    augment_audio_strong_noisy,
    augment_audio_phone_noisy,
)
from pipeline.config import (
    INDEX_WINDOWS_PER_SONG,
    FAISS_TOP_K,
    CACHE_PREPARED_IN_MEMORY,
    CACHE_PREPARED_MAX_TRACKS,
    EMB_DIM,
    FIXED_EVAL_DIR,
    SR,
)
from pipeline.dataset import (
    extract_sliding_windows,
    extract_uniform_index_windows,
    pad_or_trim,
    random_segment,
)
from pipeline.manifest import load_prepared_manifest
from pipeline.to_mel import to_mel_batch

logger = logging.getLogger("ml-pipeline.evaluate")


class PreparedAudioCache:
    def __init__(
        self,
        enabled: bool = CACHE_PREPARED_IN_MEMORY,
        max_tracks: int = CACHE_PREPARED_MAX_TRACKS,
    ):
        self.enabled = enabled
        self.max_tracks = max_tracks
        self._cache: dict[str, np.ndarray] = {}

    def load(self, path: str) -> np.ndarray:
        if not self.enabled:
            return np.load(path).astype("float32")

        cached = self._cache.get(path)
        if cached is not None:
            return cached

        audio = np.load(path).astype("float32")

        if self.max_tracks <= 0 or len(self._cache) < self.max_tracks:
            self._cache[path] = audio

        return audio


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed_windows(
    model: AudioEncoder,
    windows: list[np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    if not windows:
        return np.empty((0, EMB_DIM), dtype="float32")

    all_embs: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            chunk = windows[start:start + batch_size]
            audio_batch = np.stack(chunk).astype("float32")

            mel_batch = to_mel_batch(
                audio_batch,
                device=device,
                normalize=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                embs = model(mel_batch)

            all_embs.append(embs.float().detach().cpu().numpy().astype("float32"))

    result = np.concatenate(all_embs, axis=0)
    faiss.normalize_L2(result)
    return result


def build_eval_index(
    model: AudioEncoder,
    items: list[dict],
    cache: PreparedAudioCache,
    device: torch.device,
    windows_per_song: int,
    embed_batch_size: int = 64,
) -> tuple[faiss.Index, np.ndarray]:
    all_embeddings: list[np.ndarray] = []
    all_song_ids: list[int] = []

    pending_windows: list[np.ndarray] = []
    pending_song_ids: list[int] = []

    timings = defaultdict(float)
    flush_count = 0

    def flush_pending() -> None:
        nonlocal pending_windows, pending_song_ids, flush_count

        if not pending_windows:
            return

        t0 = time.perf_counter()

        embs = embed_windows(
            model=model,
            windows=pending_windows,
            device=device,
            batch_size=embed_batch_size,
        )

        timings["flush_embed"] += time.perf_counter() - t0
        flush_count += 1

        all_embeddings.append(embs)
        all_song_ids.extend(pending_song_ids)

        pending_windows = []
        pending_song_ids = []

    for idx, row in enumerate(items, start=1):
        t0 = time.perf_counter()
        audio = cache.load(row["prepared_path"])
        timings["load_audio"] += time.perf_counter() - t0

        track_id = int(row["track_id"])

        t0 = time.perf_counter()
        windows = extract_uniform_index_windows(audio, n_windows=windows_per_song)
        timings["extract_windows"] += time.perf_counter() - t0

        pending_windows.extend(windows)
        pending_song_ids.extend([track_id] * len(windows))

        if len(pending_windows) >= embed_batch_size:
            flush_pending()

        if idx % 250 == 0 or idx == len(items):
            total_vectors = sum(len(x) for x in all_embeddings)
            logger.info(
                "Eval index build progress processed=%s/%s vectors=%s pending=%s timings=%s flush_count=%s",
                idx,
                len(items),
                total_vectors,
                len(pending_windows),
                {k: round(v, 2) for k, v in timings.items()},
                flush_count,
            )

    flush_pending()

    embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
    song_ids = np.asarray(all_song_ids, dtype=np.int64)

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError(f"Invalid eval embeddings shape: {embeddings.shape}")

    if len(embeddings) != len(song_ids):
        raise ValueError(
            f"embeddings/song_ids mismatch: embeddings={len(embeddings)} song_ids={len(song_ids)}"
        )

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    logger.info(
        "Eval index built tracks=%s vectors=%s dim=%s windows_per_song=%s",
        len(items),
        index.ntotal,
        index.d,
        windows_per_song,
    )

    return index, song_ids


def make_query_audio(audio: np.ndarray, use_full_audio: bool) -> np.ndarray:
    if use_full_audio:
        return audio.astype("float32")

    return pad_or_trim(random_segment(audio))


def load_or_create_fixed_eval_items(
    val_items: list[dict],
    eval_query_limit: int,
    fixed_eval_set_name: str,
) -> list[dict]:
    FIXED_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXED_EVAL_DIR / f"{fixed_eval_set_name}.json"

    by_track_id = {int(row["track_id"]): row for row in val_items}

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            track_ids = json.load(f)

        fixed_items = [
            by_track_id[int(track_id)]
            for track_id in track_ids
            if int(track_id) in by_track_id
        ]

        if len(fixed_items) < 5:
            raise ValueError(
                f"Fixed eval set {path} has too few valid tracks after manifest filtering: {len(fixed_items)}"
            )

        logger.info(
            "Loaded fixed eval set name=%s path=%s requested=%s valid=%s",
            fixed_eval_set_name,
            path,
            len(track_ids),
            len(fixed_items),
        )
        return fixed_items

    selected = val_items[:eval_query_limit]
    track_ids = [int(row["track_id"]) for row in selected]

    with path.open("w", encoding="utf-8") as f:
        json.dump(track_ids, f, ensure_ascii=False, indent=2)

    logger.info(
        "Created fixed eval set name=%s path=%s tracks=%s",
        fixed_eval_set_name,
        path,
        len(track_ids),
    )

    return selected


def resolve_eval_cases(eval_noise_mode: str) -> list[str]:
    mode = (eval_noise_mode or "noisy").lower().strip()

    if mode == "clean":
        return ["clean"]

    if mode == "noisy":
        return ["noisy"]

    if mode == "phone_noisy":
        return ["phone_noisy"]

    if mode == "both":
        return ["clean", "noisy"]

    if mode == "all":
        return ["clean", "noisy", "phone_noisy"]

    raise ValueError(
        f"Unsupported eval_noise_mode={eval_noise_mode}. "
        "Expected one of: clean, noisy, phone_noisy, both, all"
    )


def resolve_primary_case(eval_noise_mode: str, available_cases: list[str]) -> str:
    mode = (eval_noise_mode or "noisy").lower().strip()

    if mode in {"clean", "noisy", "phone_noisy"}:
        return mode

    if mode == "both":
        return "noisy"

    if mode == "all":
        return "phone_noisy"

    if available_cases:
        return available_cases[-1]

    raise ValueError("No available eval cases")


def make_eval_query_audio(clean_query: np.ndarray, case: str) -> np.ndarray:
    case = case.lower().strip()

    if case == "clean":
        return clean_query.astype(np.float32)

    if case == "noisy":
        return augment_audio_strong_noisy(clean_query).astype(np.float32)

    if case == "phone_noisy":
        return augment_audio_phone_noisy(clean_query).astype(np.float32)

    raise ValueError(f"Unsupported eval case={case}")


def evaluate_case(
    *,
    model: AudioEncoder,
    index: faiss.Index,
    song_ids: np.ndarray,
    val_items: list[dict],
    cache: PreparedAudioCache,
    device: torch.device,
    use_full_query_audio: bool,
    case_name: str,
    aggregation_strategy: str = "current",
    results_jsonl_path: Path | None = None,
) -> dict:
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0
    latencies_ms: list[float] = []
    result_rows: list[dict] = []

    for row in val_items:
        query_started = time.perf_counter()

        audio = cache.load(row["prepared_path"])
        clean_query = make_query_audio(audio, use_full_audio=use_full_query_audio)
        query_audio = make_eval_query_audio(clean_query=clean_query, case=case_name)

        if total < 5:
            logger.info(
                (
                    "Eval query length case=%s track_id=%s full_audio_sec=%.2f "
                    "clean_query_sec=%.2f query_sec=%.2f query_samples=%s "
                    "use_full_query_audio=%s"
                ),
                case_name,
                row["track_id"],
                len(audio) / SR,
                len(clean_query) / SR,
                len(query_audio) / SR,
                len(query_audio),
                use_full_query_audio,
            )

        query_windows = extract_sliding_windows(query_audio)

        if not query_windows:
            logger.warning(
                "Eval query skipped no_windows case=%s track_id=%s query_samples=%s",
                case_name,
                row["track_id"],
                len(query_audio),
            )
            continue

        query_embeddings = embed_windows(model, query_windows, device)

        if len(query_embeddings) == 0:
            logger.warning(
                "Eval query skipped no_embeddings case=%s track_id=%s windows=%s",
                case_name,
                row["track_id"],
                len(query_windows),
            )
            continue

        scores, indices = index.search(query_embeddings, k=min(FAISS_TOP_K, index.ntotal))

        result = aggregate_results(
            scores=scores,
            indices=indices,
            song_ids=song_ids,
            strategy=aggregation_strategy,
        )

        latency_ms = (time.perf_counter() - query_started) * 1000.0
        latencies_ms.append(latency_ms)

        gt = int(row["track_id"])
        total += 1

        predicted_song_id = None
        ranked: list[int] = []
        correct_top1 = False
        correct_top5 = False

        confidence = None
        margin = None
        score = None
        support = None
        support_ratio = None
        support_gap = None
        support_ratio_gap = None
        hit_count = None
        hit_count_gap = None
        top_candidates = []

        if result is not None:
            predicted_song_id = int(result["song_id"])
            top_candidates = result.get("top_candidates", [])
            ranked = [int(x["song_id"]) for x in top_candidates]

            correct_top1 = bool(ranked and ranked[0] == gt)
            correct_top5 = bool(gt in ranked[:5])

            confidence = result.get("confidence")
            margin = result.get("margin")
            score = result.get("score")
            support = result.get("support")
            support_ratio = result.get("support_ratio")
            support_gap = result.get("support_gap")
            support_ratio_gap = result.get("support_ratio_gap")
            hit_count = result.get("hit_count")
            hit_count_gap = result.get("hit_count_gap")

            if correct_top1:
                correct_at_1 += 1

            if correct_top5:
                correct_at_5 += 1

        result_rows.append(
            {
                "track_id": gt,
                "split": row.get("split"),
                "case": case_name,
                "aggregation_strategy": aggregation_strategy,
                "is_positive": True,
                "predicted_song_id": predicted_song_id,
                "correct_top1": correct_top1,
                "correct_top5": correct_top5,
                "score": score,
                "confidence": confidence,
                "margin": margin,
                "support": support,
                "support_ratio": support_ratio,
                "support_gap": support_gap,
                "support_ratio_gap": support_ratio_gap,
                "hit_count": hit_count,
                "hit_count_gap": hit_count_gap,
                "latency_ms": latency_ms,
                "query_windows": len(query_windows),
                "query_sec": len(query_audio) / SR,
                "top_candidates": top_candidates,
            }
        )

    if results_jsonl_path is not None and result_rows:
        append_jsonl(results_jsonl_path, result_rows)

    lat = np.asarray(latencies_ms, dtype=np.float32)

    return {
        "count": total,
        "recall_at_1": correct_at_1 / total if total else 0.0,
        "recall_at_5": correct_at_5 / total if total else 0.0,
        "latency_p50_ms": float(np.percentile(lat, 50)) if len(lat) else None,
        "latency_p95_ms": float(np.percentile(lat, 95)) if len(lat) else None,
    }


def evaluate_prepared(
    model_path: str | Path,
    output_metrics_path: str | Path,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
    eval_query_limit: int | None = None,
    index_windows_override: int | None = None,
    use_full_query_audio: bool = False,
    fixed_eval_set_name: str = "default",
    eval_noise_mode: str = "both",
    aggregation_strategies: list[str] | None = None,
    results_jsonl_path: str | Path | None = None,
) -> dict:
    started = time.time()

    train_limit = train_limit or 0
    val_limit_for_index = val_limit or 0
    test_limit = test_limit or 0
    eval_query_limit = eval_query_limit or val_limit_for_index or 150

    index_items = load_prepared_manifest(
        train_limit=train_limit,
        val_limit=val_limit_for_index,
        test_limit=test_limit,
    )

    if not index_items:
        raise ValueError("No prepared items for eval index")

    val_items = [row for row in index_items if row["split"] == "val"]

    if val_limit is not None and val_limit > 0:
        val_items = val_items[:val_limit]

    if len(val_items) < 5:
        raise ValueError(f"Need at least 5 validation queries, got {len(val_items)}")

    val_items = load_or_create_fixed_eval_items(
        val_items=val_items,
        eval_query_limit=eval_query_limit,
        fixed_eval_set_name=fixed_eval_set_name,
    )

    windows_per_song = (
        index_windows_override
        if index_windows_override is not None
        else INDEX_WINDOWS_PER_SONG
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    eval_cases = resolve_eval_cases(eval_noise_mode)
    primary_case_name = resolve_primary_case(eval_noise_mode, eval_cases)

    logger.info(
        (
            "Evaluate device check cuda_available=%s device=%s gpu=%s "
            "fixed_eval_set=%s noise_mode=%s cases=%s primary_case=%s"
        ),
        torch.cuda.is_available(),
        device,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        fixed_eval_set_name,
        eval_noise_mode,
        eval_cases,
        primary_case_name,
    )

    cache = PreparedAudioCache()

    model = AudioEncoder().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    index, song_ids = build_eval_index(
        model=model,
        items=index_items,
        cache=cache,
        device=device,
        windows_per_song=windows_per_song,
    )

    output_metrics_path = Path(output_metrics_path)

    if results_jsonl_path is None:
        results_jsonl_path = output_metrics_path.parent / "eval_results.jsonl"
    else:
        results_jsonl_path = Path(results_jsonl_path)

    if results_jsonl_path.exists():
        results_jsonl_path.unlink()

    if aggregation_strategies is None or not aggregation_strategies:
        aggregation_strategies = ["current"]

    cases: dict[str, dict[str, dict]] = {}

    for strategy in aggregation_strategies:
        strategy_cases: dict[str, dict] = {}

        for case_name in eval_cases:
            strategy_cases[case_name] = evaluate_case(
                model=model,
                index=index,
                song_ids=song_ids,
                val_items=val_items,
                cache=cache,
                device=device,
                use_full_query_audio=use_full_query_audio,
                case_name=case_name,
                aggregation_strategy=strategy,
                results_jsonl_path=results_jsonl_path,
            )

        cases[strategy] = strategy_cases

    best_strategy = None
    best_case_metrics = None
    best_score = float("-inf")

    for strategy, strategy_cases in cases.items():
        case_metrics = strategy_cases.get(primary_case_name)

        if case_metrics is None:
            continue

        score = float(case_metrics.get("recall_at_1", 0.0))

        if score > best_score:
            best_score = score
            best_strategy = strategy
            best_case_metrics = case_metrics

    if best_strategy is None or best_case_metrics is None:
        best_strategy = aggregation_strategies[0]
        primary_case_name = next(iter(cases[best_strategy].keys()))
        best_case_metrics = cases[best_strategy][primary_case_name]

    metrics = {
        **best_case_metrics,
        "cases": cases,
        "primary_strategy": best_strategy,
        "primary_case": primary_case_name,
        "aggregation_strategies": aggregation_strategies,
        "eval_results_path": str(results_jsonl_path),
        "train_limit": train_limit,
        "val_limit": val_limit,
        "test_limit": test_limit,
        "eval_query_limit": eval_query_limit,
        "fixed_eval_set_name": fixed_eval_set_name,
        "eval_noise_mode": eval_noise_mode,
        "eval_cases": eval_cases,
        "index_tracks": len(index_items),
        "index_vectors": int(index.ntotal),
        "index_windows_per_song": windows_per_song,
        "use_full_query_audio": use_full_query_audio,
        "elapsed_sec": time.time() - started,
    }

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with output_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Evaluation finished metrics=%s output=%s", metrics, output_metrics_path)
    return metrics