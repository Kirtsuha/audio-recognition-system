import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from app.inference import aggregate_results
from app.model import AudioEncoder
from pipeline.augment import augment_audio
from pipeline.config import (
    INDEX_WINDOWS_PER_SONG,
    FAISS_TOP_K,
    CACHE_PREPARED_IN_MEMORY,
    CACHE_PREPARED_MAX_TRACKS, EMB_DIM,
)
from pipeline.dataset import (
    extract_sliding_windows,
    extract_uniform_index_windows,
    pad_or_trim,
    random_segment,
)
from pipeline.manifest import iter_prepared_manifest, load_prepared_manifest
from pipeline.to_mel import to_mel_batch

from collections import defaultdict

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


def embed_windows(
    model: AudioEncoder,
    windows: list[np.ndarray],
    device: torch.device,
    batch_size: int = 256,
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

            all_embs.append(
                embs.float().detach().cpu().numpy().astype("float32")
            )

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
        nonlocal pending_windows, pending_song_ids

        # logger.info("Eval flush start windows=%s batch_size=%s", len(pending_windows), embed_batch_size)
        if not pending_windows:
            return

        embs = embed_windows(
            model=model,
            windows=pending_windows,
            device=device,
            batch_size=embed_batch_size,
        )
        # logger.info("Eval flush finished embeddings=%s", embs.shape[0])

        all_embeddings.append(embs)
        all_song_ids.extend(pending_song_ids)

        pending_windows = []
        pending_song_ids = []

    for idx, row in enumerate(items, start=1):
        audio = cache.load(row["prepared_path"])
        track_id = int(row["track_id"])

        windows = extract_uniform_index_windows(
            audio,
            n_windows=windows_per_song,
        )

        pending_windows.extend(windows)
        pending_song_ids.extend([track_id] * len(windows))

        if len(pending_windows) >= embed_batch_size:
            flush_pending()

        if idx % 250 == 0 or idx == len(items):
            logger.info(
                "Eval index build progress processed=%s/%s vectors=%s pending=%s",
                idx,
                len(items),
                sum(len(x) for x in all_embeddings),
                len(pending_windows),
            )

    flush_pending()

    for idx, row in enumerate(items, start=1):
        t0 = time.perf_counter()
        audio = cache.load(row["prepared_path"])
        timings["load_audio"] += time.perf_counter() - t0

        track_id = int(row["track_id"])

        t0 = time.perf_counter()
        windows = extract_uniform_index_windows(
            audio,
            n_windows=windows_per_song,
        )
        timings["extract_windows"] += time.perf_counter() - t0

        pending_windows.extend(windows)
        pending_song_ids.extend([track_id] * len(windows))

        if len(pending_windows) >= embed_batch_size:
            t0 = time.perf_counter()
            flush_pending()
            timings["flush_embed"] += time.perf_counter() - t0
            flush_count += 1

        if idx % 250 == 0 or idx == len(items):
            total_vectors = sum(len(x) for x in all_embeddings)
            logger.info(
                "Eval index progress processed=%s/%s vectors=%s pending=%s timings=%s flush_count=%s",
                idx,
                len(items),
                total_vectors,
                len(pending_windows),
                {k: round(v, 2) for k, v in timings.items()},
                flush_count,
            )

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


def evaluate_prepared(
    model_path: str | Path,
    output_metrics_path: str | Path,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
    eval_query_limit: int | None = None,
    index_windows_override: int | None = None,
    use_full_query_audio: bool = False,
) -> dict:
    started = time.time()

    train_limit = train_limit or 0
    val_limit_for_index = val_limit or 0
    test_limit = test_limit or 0

    index_items = load_prepared_manifest(
        train_limit=train_limit,
        val_limit=val_limit_for_index,
        test_limit=test_limit,
    )

    if not index_items:
        raise ValueError("No prepared items for eval index")

    val_items = [row for row in index_items if row["split"] == "val"]

    if val_limit is not None:
        val_items = val_items[:val_limit]

    if eval_query_limit is not None:
        val_items = val_items[:eval_query_limit]

    if len(val_items) < 5:
        raise ValueError(f"Need at least 5 validation queries, got {len(val_items)}")

    windows_per_song = index_windows_override if index_windows_override is not None else INDEX_WINDOWS_PER_SONG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Evaluate device check cuda_available=%s device=%s gpu=%s",
        torch.cuda.is_available(),
        device,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
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

    correct_at_1 = 0
    correct_at_5 = 0
    total = 0

    latencies_ms: list[float] = []

    for row in val_items:
        query_started = time.perf_counter()

        audio = cache.load(row["prepared_path"])
        query_audio = make_query_audio(audio, use_full_audio=use_full_query_audio)
        query_audio = augment_audio(query_audio)

        query_windows = extract_sliding_windows(query_audio)
        query_embeddings = embed_windows(model, query_windows, device)

        scores, indices = index.search(query_embeddings, k=min(FAISS_TOP_K, index.ntotal))
        result = aggregate_results(scores=scores, indices=indices, song_ids=song_ids)

        latency_ms = (time.perf_counter() - query_started) * 1000.0
        latencies_ms.append(latency_ms)

        gt = int(row["track_id"])
        total += 1

        if result is None:
            continue

        ranked = [int(x["song_id"]) for x in result.get("top_candidates", [])]

        if ranked and ranked[0] == gt:
            correct_at_1 += 1

        if gt in ranked[:5]:
            correct_at_5 += 1

    lat = np.asarray(latencies_ms, dtype=np.float32)

    metrics = {
        "count": total,
        "recall_at_1": correct_at_1 / total if total else 0.0,
        "recall_at_5": correct_at_5 / total if total else 0.0,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "test_limit": test_limit,
        "eval_query_limit": eval_query_limit,
        "index_tracks": len(index_items),
        "index_vectors": int(index.ntotal),
        "index_windows_per_song": windows_per_song,
        "use_full_query_audio": use_full_query_audio,
        "latency_p50_ms": float(np.percentile(lat, 50)) if len(lat) else None,
        "latency_p95_ms": float(np.percentile(lat, 95)) if len(lat) else None,
        "elapsed_sec": time.time() - started,
    }

    output_metrics_path = Path(output_metrics_path)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with output_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Evaluation finished metrics=%s output=%s", metrics, output_metrics_path)
    return metrics