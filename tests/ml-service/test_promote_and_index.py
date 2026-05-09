import json

import faiss
import numpy as np
import pytest

import pipeline.promote as promote
from pipeline.build_index import append_to_faiss_index, build_faiss_index


def test_promote_run_to_active(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    active_dir = tmp_path / "active"
    run_dir.mkdir()

    required = [
        "model.pt",
        "embeddings.npy",
        "song_ids.npy",
        "faiss.index",
        "song_manifest.json",
        "metrics.json",
    ]

    for name in required:
        (run_dir / name).write_bytes(b"data")

    monkeypatch.setattr(promote, "ACTIVE_ARTIFACTS_DIR", active_dir)

    promote.promote_run_to_active(run_dir)

    for name in required:
        assert (active_dir / name).exists()

    pointer = json.loads((active_dir / "active_run.json").read_text(encoding="utf-8"))
    assert pointer["run_dir"] == str(run_dir)


def test_promote_missing_artifact_raises(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    active_dir = tmp_path / "active"
    run_dir.mkdir()

    monkeypatch.setattr(promote, "ACTIVE_ARTIFACTS_DIR", active_dir)

    with pytest.raises(FileNotFoundError):
        promote.promote_run_to_active(run_dir)


def test_build_faiss_index(tmp_path, monkeypatch):
    work = tmp_path / "build_faiss_index"
    work.mkdir()

    embeddings_path = work / "embeddings.npy"
    song_ids_path = work / "song_ids.npy"
    index_path = work / "faiss.index"

    np.save(
        embeddings_path,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.save(song_ids_path, np.array([10, 20], dtype=np.int64))

    written = {}

    def fake_write_index(index, path):
        written["index"] = index
        written["path"] = path

    monkeypatch.setattr(faiss, "write_index", fake_write_index)

    summary = build_faiss_index(
        embeddings_path,
        song_ids_path,
        index_path,
        index_type="flat",
    )

    index = written["index"]
    assert index.ntotal == 2
    assert index.d == 2
    assert written["path"] == str(index_path)
    assert summary["vectors"] == 2


def test_build_faiss_index_invalid_shape(tmp_path):
    embeddings_path = tmp_path / "embeddings.npy"
    song_ids_path = tmp_path / "song_ids.npy"
    index_path = tmp_path / "faiss.index"

    np.save(embeddings_path, np.array([1, 2, 3], dtype=np.float32))
    np.save(song_ids_path, np.array([1, 2, 3], dtype=np.int64))

    with pytest.raises(ValueError):
        build_faiss_index(embeddings_path, song_ids_path, index_path)


def test_build_faiss_index_length_mismatch(tmp_path):
    embeddings_path = tmp_path / "embeddings.npy"
    song_ids_path = tmp_path / "song_ids.npy"
    index_path = tmp_path / "faiss.index"

    np.save(embeddings_path, np.ones((2, 3), dtype=np.float32))
    np.save(song_ids_path, np.array([1], dtype=np.int64))

    with pytest.raises(ValueError):
        build_faiss_index(embeddings_path, song_ids_path, index_path)


def test_append_to_faiss_index(tmp_path, monkeypatch):
    work = tmp_path / "append_faiss_index"
    work.mkdir()

    base_index_path = work / "base.index"
    new_embeddings_path = work / "new_embeddings.npy"
    output_index_path = work / "out.index"

    base_vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    faiss.normalize_L2(base_vectors)

    index = faiss.IndexFlatIP(2)
    index.add(base_vectors)
    np.save(new_embeddings_path, np.array([[0.0, 1.0]], dtype=np.float32))

    written = {}
    monkeypatch.setattr(faiss, "read_index", lambda path: index)
    monkeypatch.setattr(faiss, "write_index", lambda index, path: written.update(index=index, path=path))

    append_to_faiss_index(base_index_path, new_embeddings_path, output_index_path)

    assert index.ntotal == 2
    assert written["index"] is index
    assert written["path"] == str(output_index_path)
