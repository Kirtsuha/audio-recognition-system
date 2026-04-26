from types import SimpleNamespace

import numpy as np
import pytest

import app.runtime as runtime


def reset_runtime_state():
    runtime._model = None
    runtime._index = None
    runtime._song_ids = None


def test_runtime_not_ready_raises():
    reset_runtime_state()

    assert runtime.runtime_ready() is False

    with pytest.raises(RuntimeError):
        runtime.ensure_runtime_ready()


def test_get_runtime_when_ready():
    fake_model = SimpleNamespace(head=[None, SimpleNamespace(out_features=128)])
    fake_index = SimpleNamespace(ntotal=3, d=128)
    fake_song_ids = np.array([1, 2, 3], dtype=np.int64)

    runtime._model = fake_model
    runtime._index = fake_index
    runtime._song_ids = fake_song_ids

    artifacts = runtime.get_runtime()

    assert artifacts.model is fake_model
    assert artifacts.index is fake_index
    assert artifacts.song_ids.tolist() == [1, 2, 3]


def test_should_accept_true_and_false(monkeypatch):
    monkeypatch.setattr(runtime, "MIN_CONFIDENCE", 0.5)
    monkeypatch.setattr(runtime, "MIN_MARGIN", 0.1)
    monkeypatch.setattr(runtime, "MIN_SUPPORTED_WINDOWS", 2)

    assert runtime.should_accept(
        {"confidence": 0.6, "margin": 0.2, "support": 2}
    ) is True

    assert runtime.should_accept(
        {"confidence": 0.4, "margin": 0.2, "support": 2}
    ) is False


def test_runtime_status_when_empty(tmp_path, monkeypatch):
    reset_runtime_state()

    monkeypatch.setattr(runtime, "MODEL_PATH", tmp_path / "missing-model.pt")
    monkeypatch.setattr(runtime, "FAISS_INDEX_PATH", tmp_path / "missing.index")
    monkeypatch.setattr(runtime, "SONG_IDS_PATH", tmp_path / "missing-song-ids.npy")

    status = runtime.runtime_status()

    assert status["runtime_ready"] is False
    assert status["model_exists"] is False
    assert status["index_total"] is None


def test_load_runtime_artifacts_missing_files_degraded(tmp_path, monkeypatch):
    reset_runtime_state()

    monkeypatch.setattr(runtime, "MODEL_PATH", tmp_path / "model.pt")
    monkeypatch.setattr(runtime, "FAISS_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr(runtime, "SONG_IDS_PATH", tmp_path / "song_ids.npy")

    runtime.load_runtime_artifacts()

    assert runtime.runtime_ready() is False