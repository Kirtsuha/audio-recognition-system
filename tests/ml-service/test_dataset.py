import random

import numpy as np

from pipeline import dataset


def test_pad_or_trim_trims():
    audio = np.arange(10, dtype=np.float32)

    result = dataset.pad_or_trim(audio, target_len=5)

    assert result.tolist() == [0, 1, 2, 3, 4]
    assert result.dtype == np.float32


def test_pad_or_trim_pads():
    audio = np.array([1, 2, 3], dtype=np.float32)

    result = dataset.pad_or_trim(audio, target_len=5)

    assert result.tolist() == [1, 2, 3, 0, 0]


def test_random_segment_short_audio_pads():
    audio = np.array([1, 2], dtype=np.float32)

    result = dataset.random_segment(audio, target_len=5)

    assert len(result) == 5
    assert result[:2].tolist() == [1, 2]


def test_random_segment_long_audio(monkeypatch):
    monkeypatch.setattr(random, "randint", lambda a, b: 2)

    audio = np.arange(10, dtype=np.float32)
    result = dataset.random_segment(audio, target_len=4)

    assert result.tolist() == [2, 3, 4, 5]


def test_extract_sliding_windows_short_audio():
    audio = np.ones(5, dtype=np.float32)

    windows = dataset.extract_sliding_windows(
        audio,
        window_seconds=1,
        hop_seconds=1,
        max_windows=10,
    )

    assert len(windows) == 1
    assert len(windows[0]) == dataset.SR


def test_extract_sliding_windows_limits_max_windows():
    audio = np.ones(dataset.SR * 10, dtype=np.float32)

    windows = dataset.extract_sliding_windows(
        audio,
        window_seconds=1,
        hop_seconds=1,
        max_windows=3,
    )

    assert len(windows) == 3
    assert all(len(w) == dataset.SR for w in windows)


def test_extract_uniform_index_windows():
    audio = np.arange(dataset.SR * 5, dtype=np.float32)

    windows = dataset.extract_uniform_index_windows(
        audio,
        n_windows=3,
        window_seconds=1,
    )

    assert len(windows) == 3
    assert all(len(w) == dataset.SR for w in windows)


def test_assign_split_is_stable():
    assert dataset.assign_split(123) == dataset.assign_split(123)
    assert dataset.assign_split(123) in {"train", "val", "test"}