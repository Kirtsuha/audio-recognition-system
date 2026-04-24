import json
import logging
import random
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
    CACHE_PREPARED_MAX_TRACKS,
)
from pipeline.dataset import (
    extract_sliding_windows,
    extract_uniform_index_windows,
    iter_prepared_manifest,
    load_prepared_manifest,
    pad_or_trim,
    random_segment,
)
from pipeline.to_mel import to_mel

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
) -> np.ndarray:
    batch = torch.stack([to_mel(w) for w in windows]).to(device)

    with torch.inference_mode():
        embs = model(batch).cpu().numpy().astype("float32")

    faiss.normalize_L2(embs)
    return embs


def build_eval_index(
    model: AudioEncoder,
    items: list[dict],
    cache: PreparedAudioCache,
    device: torch.device,
    windows_per_song: int,
) -> tuple[faiss.Index, np.ndarray]:
    all_embeddings: list[np.ndarray] = []
    all_song_ids: list[int] = []

    for idx, row in enumerate(items, start=1):
        audio = cache.load(row["prepared_path"])

        windows = extract_uniform_index_windows(
            audio,
            n_windows=windows_per_song,
        )

        embs = embed_windows(model, windows, device)

        for emb in embs:
            all_embeddings.append(emb)
            all_song_ids.append(int(row["track_id"]))

        if idx % 250 == 0 or idx == len(items):
            logger.info(
                "Eval index build progress processed=%s/%s vectors=%s",
                idx,
                len(items),
                len(all_embeddings),
            )

    embeddings = np.asarray(all_embeddings, dtype="float32")
    song_ids = np.asarray(all_song_ids, dtype=np.int64)

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError(f"Invalid eval embeddings shape: {embeddings.shape}")

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