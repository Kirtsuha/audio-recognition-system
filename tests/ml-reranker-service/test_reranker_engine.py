from types import SimpleNamespace

import numpy as np
import pytest

from app.reranker import RerankerEngine
from app.schemas import FingerprintCandidate, FingerprintRecognitionPayload, RerankOptions


def make_engine(monkeypatch):
    monkeypatch.setattr(RerankerEngine, "load_model", lambda self: setattr(self, "model", object()))
    monkeypatch.setattr(RerankerEngine, "load_reference_store", lambda self: setattr(self, "reference_store", None))
    return RerankerEngine(storage=SimpleNamespace(get_bytes=lambda bucket, key: b"audio"))


def test_status_and_option_helpers(monkeypatch):
    engine = make_engine(monkeypatch)
    monkeypatch.setattr("app.reranker.settings.offset_jitter_seconds", "-1,0,1")

    assert engine.status().model_loaded is True
    assert engine._parse_default_jitter() == [-1.0, 0.0, 1.0]
    assert engine._option_weights(RerankOptions(fp_weight=0, ml_weight=0)) == (1.0, 0.0)
    assert engine._similarity_to_probability(-1.0) == 0.0
    assert engine._similarity_to_probability(1.0) == 1.0


def test_ensure_ready_raises(monkeypatch):
    engine = make_engine(monkeypatch)
    engine.model = None

    with pytest.raises(RuntimeError):
        engine.ensure_ready()


def test_rerank_no_candidates(monkeypatch):
    engine = make_engine(monkeypatch)
    monkeypatch.setattr(engine, "_load_audio_from_s3", lambda bucket, key: np.ones(10, dtype=np.float32))

    response = engine.rerank(
        request_id="req",
        query_bucket="queries",
        query_key="q.wav",
        reference_bucket="tracks",
        fingerprint=FingerprintRecognitionPayload(candidates=[]),
        options=RerankOptions(),
    )

    assert response.matched is False
    assert response.reason == "no_fingerprint_candidates"


def test_rerank_scores_precomputed_candidate(monkeypatch):
    engine = make_engine(monkeypatch)
    candidate = FingerprintCandidate(
        track_id=1,
        best_offset=0,
        aligned_matches=10,
        total_matches=12,
        offset_count=1,
        coverage=0.5,
        unique_coverage=0.5,
        score_gap=4.0,
        confidence=0.8,
        title="Song",
        artist="Artist",
        s3_key="song.wav",
    )

    monkeypatch.setattr(engine, "_load_audio_from_s3", lambda bucket, key: np.ones(20, dtype=np.float32))
    monkeypatch.setattr(engine, "_embed_batch", lambda segments: np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(
        engine,
        "_score_candidate_precomputed",
        lambda candidate, query_embedding, offset_sec: (0.9, 1.5),
    )

    response = engine.rerank(
        request_id="req",
        query_bucket="queries",
        query_key="q.wav",
        reference_bucket="tracks",
        fingerprint=FingerprintRecognitionPayload(candidates=[candidate]),
        options=RerankOptions(no_match_threshold=0.1, fp_weight=0.5, ml_weight=0.5),
    )

    assert response.matched is True
    assert response.best.track_id == 1
    assert response.best.best_offset_sec == 1.5
    assert response.best.ml_probability == pytest.approx(0.95)
