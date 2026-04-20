import random
from typing import List

import librosa
import numpy as np

from pipeline.config import SR, SEGMENT_SECONDS, MAX_QUERY_WINDOWS, WINDOW_HOP_SECONDS


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio.astype(np.float32)


def segment_num_samples() -> int:
    return int(SEGMENT_SECONDS * SR)


def pad_or_trim(audio: np.ndarray, target_len: int | None = None) -> np.ndarray:
    if target_len is None:
        target_len = segment_num_samples()

    if len(audio) == target_len:
        return audio.astype(np.float32)

    if len(audio) > target_len:
        return audio[:target_len].astype(np.float32)

    out = np.zeros(target_len, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def random_segment(audio: np.ndarray, target_len: int | None = None) -> np.ndarray:
    if target_len is None:
        target_len = segment_num_samples()

    if len(audio) <= target_len:
        return pad_or_trim(audio, target_len)

    max_start = len(audio) - target_len
    start = random.randint(0, max_start)
    end = start + target_len
    return audio[start:end].astype(np.float32)


def extract_sliding_windows(
    audio: np.ndarray,
    window_seconds: float = SEGMENT_SECONDS,
    hop_seconds: float = WINDOW_HOP_SECONDS,
    max_windows: int = MAX_QUERY_WINDOWS,
) -> List[np.ndarray]:
    window_len = int(window_seconds * SR)
    hop_len = int(hop_seconds * SR)

    if len(audio) <= window_len:
        return [pad_or_trim(audio, window_len)]

    starts = list(range(0, len(audio) - window_len + 1, hop_len))
    if not starts:
        return [pad_or_trim(audio, window_len)]

    if len(starts) > max_windows:
        chosen = np.linspace(0, len(starts) - 1, max_windows).astype(int)
        starts = [starts[i] for i in chosen]

    return [audio[s : s + window_len].astype(np.float32) for s in starts]


def extract_uniform_index_windows(audio: np.ndarray, n_windows: int, window_seconds: float = SEGMENT_SECONDS) -> List[np.ndarray]:
    window_len = int(window_seconds * SR)

    if len(audio) <= window_len:
        return [pad_or_trim(audio, window_len)]

    max_start = len(audio) - window_len
    starts = np.linspace(0, max_start, n_windows).astype(int)
    return [audio[s : s + window_len].astype(np.float32) for s in starts]