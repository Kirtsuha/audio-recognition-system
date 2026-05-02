import random
import tempfile
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf

from repository.s3_client import get_s3


def load_s3_audio(
    bucket: str,
    key: str,
    sample_rate: int,
) -> tuple[np.ndarray, float | None]:
    s3 = get_s3()
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = BytesIO(obj["Body"].read())

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        tmp.write(data.read())
        tmp.flush()

        duration_sec = None
        try:
            info = sf.info(tmp.name)
            if info.frames and info.samplerate:
                duration_sec = float(info.frames / info.samplerate)
        except Exception:
            duration_sec = None

        audio, _ = librosa.load(tmp.name, sr=sample_rate, mono=True)

    return normalize(audio), duration_sec


def normalize(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)


def cut_random_segment(
    audio: np.ndarray,
    sample_rate: int,
    duration_sec: float,
) -> np.ndarray:
    target_len = int(sample_rate * duration_sec)

    if len(audio) <= target_len:
        out = np.zeros(target_len, dtype=np.float32)
        out[: len(audio)] = audio.astype(np.float32)
        return out

    start = random.randint(0, len(audio) - target_len)
    return audio[start:start + target_len].astype(np.float32)


def add_gaussian_noise(audio: np.ndarray, snr_db_min: float, snr_db_max: float) -> np.ndarray:
    snr_db = random.uniform(snr_db_min, snr_db_max)
    signal_power = np.mean(audio ** 2) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    noise = np.random.randn(len(audio)).astype(np.float32)
    noise = noise / (np.std(noise) + 1e-8)
    noise = noise * np.sqrt(noise_power)

    return normalize(audio + noise)


def random_gain(audio: np.ndarray, min_db: float = -10.0, max_db: float = 6.0) -> np.ndarray:
    gain_db = random.uniform(min_db, max_db)
    gain = 10 ** (gain_db / 20.0)
    return normalize(audio * gain)


def apply_corruption(audio: np.ndarray, corruption: str) -> np.ndarray:
    if corruption == "clean":
        return normalize(audio)

    if corruption == "noise_snr20":
        return add_gaussian_noise(audio, 18.0, 22.0)

    if corruption == "noise_snr10":
        return add_gaussian_noise(audio, 9.0, 11.0)

    if corruption == "gain":
        return random_gain(audio)

    raise ValueError(f"Unsupported corruption: {corruption}")