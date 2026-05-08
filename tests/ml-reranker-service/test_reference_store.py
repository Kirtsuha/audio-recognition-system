import json

import numpy as np
import pytest

from app.reference_store import ReferenceEmbeddingStore


def test_reference_store_missing_embeddings_raises(tmp_path):
    store = ReferenceEmbeddingStore(tmp_path / "missing.npy", tmp_path / "meta.jsonl")

    with pytest.raises(FileNotFoundError):
        store.load()


def test_reference_store_loads_and_finds_nearby_embeddings(tmp_path):
    embeddings_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "meta.jsonl"
    config_path = tmp_path / "config.json"

    np.save(embeddings_path, np.array([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32))
    rows = [
        {"embedding_index": 0, "track_id": 10, "s3_key": "a.wav", "start_sec": 0.0, "window_idx": 0},
        {"embedding_index": 1, "track_id": 10, "s3_key": "a.wav", "start_sec": 2.0, "window_idx": 1},
        {"embedding_index": 2, "track_id": 20, "s3_key": "b.wav", "start_sec": 0.0, "window_idx": 0},
    ]
    meta_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    config_path.write_text('{"window_seconds": 15}', encoding="utf-8")

    store = ReferenceEmbeddingStore(embeddings_path, meta_path, config_path)
    store.load()

    assert store.loaded() is True
    assert store.status()["vectors"] == 3
    embeddings, windows = store.get_nearby_embeddings(track_id=10, offset_sec=1.7, neighbor_windows=0)
    assert embeddings.shape == (1, 2)
    assert windows[0].start_sec == 2.0
    assert store.get_nearby_embeddings(track_id=999, offset_sec=0) is None
