import numpy as np
import pytest

from app.audio import crop_segment, normalize_audio, pad_or_trim, validate_audio_key


def test_validate_audio_key_rejects_unknown_suffix():
    validate_audio_key("track.wav")

    with pytest.raises(ValueError):
        validate_audio_key("track.txt")


def test_normalize_audio_clamps_nan_and_peak():
    result = normalize_audio(np.array([0.0, np.nan, 2.0, -1.0], dtype=np.float32))

    assert result.dtype == np.float32
    assert result.tolist() == [0.0, 0.0, 1.0, -0.5]


def test_pad_or_trim():
    assert pad_or_trim(np.array([1, 2, 3], dtype=np.float32), 2).tolist() == [1.0, 2.0]
    assert pad_or_trim(np.array([1], dtype=np.float32), 3).tolist() == [1.0, 0.0, 0.0]


def test_crop_segment(monkeypatch):
    from app import audio

    monkeypatch.setattr(audio.settings, "sr", 10)
    segment = crop_segment(np.arange(50, dtype=np.float32), start_sec=1.0, segment_seconds=0.5)

    assert segment.tolist() == [10, 11, 12, 13, 14]


def test_crop_segment_past_end_returns_zeros(monkeypatch):
    from app import audio

    monkeypatch.setattr(audio.settings, "sr", 10)
    segment = crop_segment(np.arange(5, dtype=np.float32), start_sec=10.0, segment_seconds=0.3)

    assert segment.tolist() == [0.0, 0.0, 0.0]
