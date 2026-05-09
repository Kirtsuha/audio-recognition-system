from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from app.config import settings

logger = logging.getLogger("ml-reranker.audio")

SUPPORTED_AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".webm",
    ".aac",
}


class AudioDecodeError(RuntimeError):
    pass


def validate_audio_key(key: str) -> None:
    suffix = Path(key).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported audio suffix={suffix}; supported={sorted(SUPPORTED_AUDIO_SUFFIXES)}")


def load_audio_from_bytes(
    data: bytes,
    sample_rate: int | None = None,
    suffix: str | None = None,
    ) -> np.ndarray:
    target_sr = sample_rate or settings.sr
    bio = io.BytesIO(data)

    try:
        audio, sr = sf.read(bio, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return normalize_audio(audio)
    except Exception:
        logger.debug("soundfile decode failed; trying librosa", exc_info=True)

    try:
        bio.seek(0)
        audio, _ = librosa.load(bio, sr=target_sr, mono=True)
        return normalize_audio(audio)
    except Exception:
        logger.debug("librosa bytes decode failed; trying temporary file", exc_info=True)

    tmp_path = None
    safe_suffix = suffix if suffix and suffix.startswith(".") else ".audio"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=safe_suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        audio, _ = librosa.load(tmp_path, sr=target_sr, mono=True)
        return normalize_audio(audio)
    except Exception as exc:
        raise AudioDecodeError("Failed to decode audio bytes") from exc
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug("Failed to remove temporary audio file path=%s", tmp_path, exc_info=True)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype="float32")

    if audio.size == 0:
        return audio

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak

    return audio.astype("float32")


def segment_num_samples(segment_seconds: float | None = None) -> int:
    seconds = settings.segment_seconds if segment_seconds is None else float(segment_seconds)
    return int(round(seconds * settings.sr))


def pad_or_trim(audio: np.ndarray, target_len: int | None = None) -> np.ndarray:
    if target_len is None:
        target_len = segment_num_samples()

    audio = np.asarray(audio, dtype="float32")

    if len(audio) == target_len:
        return audio.astype("float32")

    if len(audio) > target_len:
        return audio[:target_len].astype("float32")

    out = np.zeros(target_len, dtype="float32")
    out[: len(audio)] = audio
    return out


def crop_segment(audio: np.ndarray, start_sec: float, segment_seconds: float | None = None) -> np.ndarray:
    seconds = settings.segment_seconds if segment_seconds is None else float(segment_seconds)
    target_len = segment_num_samples(seconds)

    start_sec = max(float(start_sec), 0.0)
    start = int(round(start_sec * settings.sr))
    end = start + target_len

    if len(audio) == 0:
        return np.zeros(target_len, dtype="float32")

    if start >= len(audio):
        return np.zeros(target_len, dtype="float32")

    return pad_or_trim(audio[start:end], target_len=target_len)
