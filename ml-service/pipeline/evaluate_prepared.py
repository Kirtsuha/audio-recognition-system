import json
import logging
from pathlib import Path

import faiss
import numpy as np
import torch

from app.model import AudioEncoder
from pipeline.augment import augment_audio
from pipeline.config import INDEX_WINDOWS_PER_SONG, CACHE_PREPARED_IN_MEMORY, CACHE_PREPARED_MAX_TRACKS
from pipeline.dataset import iter_prepared_manifest, random_segment, pad_or_trim
from pipeline.to_mel import to_mel

logger = logging.getLogger("ml-pipeline.evaluate")


class PreparedAudioCache:
    def __init__(self, enabled: bool = CACHE_PREPARED_IN_MEMORY, max_tracks: int = CACHE_PREPARED_MAX_TRACKS):
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


def evaluate_prepared(
    model_path: str | Path,
    output_metrics_path: str | Path,
    val_limit: int | None = None,
    eval_query_limit: int | None = None,
    index_windows_override: int | None = None,
) -> dict:
    val_items = [row for row in iter_prepared_manifest() if row["split"] == "val"]
    if val_limit is not None:
        val_items = val_items[:val_limit]

    if len(val_items) < 5:
        raise ValueError("Need at least 5 validation items for evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = PreparedAudioCache()

    model = AudioEncoder().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    windows_per_song = index_windows_override if index_windows_override is not None else INDEX_WINDOWS_PER_SONG

    ref_embeddings = []
    ref_song_ids = []

    for row in val_items:
        audio = cache.load(row["prepared_path"])

        windows = []
        for _ in range(windows_per_song):
            windows.append(pad_or_trim(random_segment(audio)))

        batch = torch.stack([to_mel(w) for w in windows]).to(device)
        with torch.inference_mode():
            embs = model(batch).cpu().numpy().astype("float32")

        ref_embeddings.append(embs.mean(axis=0))
        ref_song_ids.append(int(row["track_id"]))

    ref_embeddings = np.asarray(ref_embeddings, dtype="float32")
    ref_song_ids = np.asarray(ref_song_ids, dtype=np.int64)
    faiss.normalize_L2(ref_embeddings)

    index = faiss.IndexFlatIP(ref_embeddings.shape[1])
    index.add(ref_embeddings)

    query_items = list(val_items)
    if eval_query_limit is not None:
        query_items = query_items[:eval_query_limit]

    correct_at_1 = 0
    correct_at_5 = 0
    total = 0

    for row in query_items:
        audio = cache.load(row["prepared_path"])
        query = pad_or_trim(augment_audio(random_segment(audio)))
        batch = torch.stack([to_mel(query)]).to(device)

        with torch.inference_mode():
            q = model(batch).cpu().numpy().astype("float32")
        faiss.normalize_L2(q)

        _, idx = index.search(q, k=min(5, len(ref_song_ids)))
        ranked = [int(ref_song_ids[i]) for i in idx[0]]
        gt = int(row["track_id"])

        total += 1
        if ranked and ranked[0] == gt:
            correct_at_1 += 1
        if gt in ranked:
            correct_at_5 += 1

    metrics = {
        "count": total,
        "recall_at_1": correct_at_1 / total if total else 0.0,
        "recall_at_5": correct_at_5 / total if total else 0.0,
        "val_limit": val_limit,
        "eval_query_limit": eval_query_limit,
        "index_windows_per_song": windows_per_song,
    }

    output_metrics_path = Path(output_metrics_path)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Evaluation finished metrics=%s output=%s", metrics, output_metrics_path)
    return metrics