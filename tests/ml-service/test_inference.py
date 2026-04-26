import numpy as np

from app.inference import aggregate_results


def test_aggregate_results_empty_returns_none():
    scores = np.array([], dtype=np.float32)
    indices = np.array([], dtype=np.int64)
    song_ids = np.array([], dtype=np.int64)

    assert aggregate_results(scores, indices, song_ids) is None


def test_aggregate_results_ignores_negative_indices():
    scores = np.array([[0.9, 0.1]], dtype=np.float32)
    indices = np.array([[-1, -1]], dtype=np.int64)
    song_ids = np.array([10, 20], dtype=np.int64)

    assert aggregate_results(scores, indices, song_ids) is None


def test_aggregate_results_ranks_best_song():
    scores = np.array(
        [
            [0.9, 0.5, 0.1],
            [0.8, 0.4, 0.2],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )
    song_ids = np.array([101, 202, 303], dtype=np.int64)

    result = aggregate_results(scores, indices, song_ids)

    assert result is not None
    assert result["song_id"] == 101
    assert result["support"] == 2
    assert result["support_ratio"] == 1.0
    assert result["confidence"] > 0
    assert len(result["top_candidates"]) == 3


def test_aggregate_results_single_candidate_margin():
    scores = np.array([[0.7]], dtype=np.float32)
    indices = np.array([[0]], dtype=np.int64)
    song_ids = np.array([777], dtype=np.int64)

    result = aggregate_results(scores, indices, song_ids)

    assert result["song_id"] == 777
    assert result["margin"] > 0
    assert result["top_candidates"][0]["song_id"] == 777