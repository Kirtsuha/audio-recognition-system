import random

import librosa
import numpy as np


def normalize_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio)) + 1e-8)
    return (audio / max_abs * peak).astype(np.float32)


def add_gaussian_noise(
    audio: np.ndarray,
    snr_db_min: float,
    snr_db_max: float,
) -> np.ndarray:
    snr_db = random.uniform(snr_db_min, snr_db_max)

    signal_power = float(np.mean(audio ** 2)) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    noise = np.random.randn(len(audio)).astype(np.float32)
    noise = noise / (np.std(noise) + 1e-8)
    noise = noise * np.sqrt(noise_power)

    return (audio + noise).astype(np.float32)


def random_gain(
    audio: np.ndarray,
    min_db: float = -12.0,
    max_db: float = 8.0,
) -> np.ndarray:
    gain_db = random.uniform(min_db, max_db)
    gain = 10 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32)


def random_clipping(
    audio: np.ndarray,
    min_clip: float = 0.55,
    max_clip: float = 0.75,
) -> np.ndarray:
    clip_val = random.uniform(min_clip, max_clip)
    return np.clip(audio, -clip_val, clip_val).astype(np.float32)


def random_pitch_shift(audio: np.ndarray, sr: int) -> np.ndarray:
    n_steps = random.uniform(-1.0, 1.0)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps).astype(np.float32)


def random_time_stretch(audio: np.ndarray) -> np.ndarray:
    rate = random.uniform(0.95, 1.05)
    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


def simple_reverb(audio: np.ndarray, sr: int) -> np.ndarray:
    delay_ms = random.choice([20, 35, 50, 70])
    decay = random.uniform(0.15, 0.45)
    delay = int(sr * delay_ms / 1000.0)

    impulse = np.zeros(delay * 3 + 1, dtype=np.float32)
    impulse[0] = 1.0
    impulse[delay] = decay
    impulse[min(delay * 2, len(impulse) - 1)] = decay * 0.5
    impulse[min(delay * 3, len(impulse) - 1)] = decay * 0.25

    return np.convolve(audio, impulse, mode="full")[: len(audio)].astype(np.float32)


def apply_corruption(audio: np.ndarray, corruption: str, sr: int) -> np.ndarray:
    out = audio.astype(np.float32).copy()

    if corruption == "clean":
        return normalize_peak(out)

    if corruption == "noise_snr20":
        out = add_gaussian_noise(out, 18.0, 22.0)

    elif corruption == "noise_snr10":
        out = add_gaussian_noise(out, 9.0, 11.0)

    elif corruption == "noise_snr5":
        out = add_gaussian_noise(out, 4.0, 6.0)

    elif corruption == "gain":
        out = random_gain(out, -12.0, 8.0)

    elif corruption == "clipping":
        out = random_clipping(out, 0.55, 0.75)

    elif corruption == "reverb":
        out = simple_reverb(out, sr)

    elif corruption == "mixed_noise_reverb":
        out = add_gaussian_noise(out, 8.0, 14.0)
        out = simple_reverb(out, sr)

    elif corruption == "pitch_shift":
        out = random_pitch_shift(out, sr)

    elif corruption == "time_stretch":
        out = random_time_stretch(out)

    elif corruption == "strong_noisy":
        if random.random() < 0.95:
            out = random_gain(out, -14.0, 10.0)
        if random.random() < 0.95:
            out = add_gaussian_noise(out, -2.0, 18.0)
        if random.random() < 0.45:
            out = random_clipping(out, 0.45, 0.90)
        if random.random() < 0.45:
            out = simple_reverb(out, sr)
        if random.random() < 0.25:
            out = add_gaussian_noise(out, 5.0, 20.0)

    else:
        raise ValueError(f"Unsupported corruption: {corruption}")

    return normalize_peak(out)