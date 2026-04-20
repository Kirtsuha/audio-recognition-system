import random
import numpy as np
import librosa

from pipeline.config import SR


def add_gaussian_noise(audio: np.ndarray, snr_db_min: float = 5.0, snr_db_max: float = 25.0) -> np.ndarray:
    snr_db = random.uniform(snr_db_min, snr_db_max)
    signal_power = np.mean(audio ** 2) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(len(audio)).astype(np.float32)
    noise = noise / (np.std(noise) + 1e-8)
    noise = noise * np.sqrt(noise_power)
    return (audio + noise).astype(np.float32)


def random_gain(audio: np.ndarray, min_db: float = -8.0, max_db: float = 6.0) -> np.ndarray:
    gain_db = random.uniform(min_db, max_db)
    gain = 10 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32)


def random_clipping(audio: np.ndarray, min_clip: float = 0.65, max_clip: float = 0.98) -> np.ndarray:
    clip_val = random.uniform(min_clip, max_clip)
    return np.clip(audio, -clip_val, clip_val).astype(np.float32)


def random_pitch_shift(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    n_steps = random.uniform(-1.0, 1.0)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps).astype(np.float32)


def random_time_stretch(audio: np.ndarray) -> np.ndarray:
    rate = random.uniform(0.95, 1.05)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return stretched.astype(np.float32)


def simple_reverb(audio: np.ndarray) -> np.ndarray:
    delay_ms = random.choice([20, 35, 50, 70])
    decay = random.uniform(0.15, 0.45)
    delay = int(SR * delay_ms / 1000.0)

    impulse = np.zeros(delay * 3 + 1, dtype=np.float32)
    impulse[0] = 1.0
    impulse[delay] = decay
    impulse[min(delay * 2, len(impulse) - 1)] = decay * 0.5
    impulse[min(delay * 3, len(impulse) - 1)] = decay * 0.25

    reverbed = np.convolve(audio, impulse, mode="full")[: len(audio)]
    return reverbed.astype(np.float32)


def normalize_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = np.max(np.abs(audio)) + 1e-8
    return (audio / max_abs * peak).astype(np.float32)


def augment_audio(audio: np.ndarray) -> np.ndarray:
    out = audio.astype(np.float32).copy()

    if random.random() < 0.75:
        out = random_gain(out)

    if random.random() < 0.70:
        out = add_gaussian_noise(out)

    if random.random() < 0.35:
        out = simple_reverb(out)

    if random.random() < 0.25:
        out = random_pitch_shift(out)

    if random.random() < 0.25:
        out = random_time_stretch(out)

    if random.random() < 0.30:
        out = random_clipping(out)

    out = normalize_peak(out)
    return out.astype(np.float32)