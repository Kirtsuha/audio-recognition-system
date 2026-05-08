import logging
import random
import tempfile

import boto3
import librosa
import numpy as np
from botocore.exceptions import ClientError

from app.config import (
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT,
    S3_PREFIX_CANDIDATES,
    S3_SECRET_KEY,
    SR,
)

logger = logging.getLogger("evaluation.audio")


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="us-east-1",
        use_ssl=False,
    )


def _candidate_s3_keys(s3_key: str) -> list[str]:
    keys = [s3_key]

    for prefix in S3_PREFIX_CANDIDATES:
        if not prefix:
            continue

        prefix = prefix.strip("/")
        clean_key = s3_key.lstrip("/")

        keys.append(f"{prefix}/{clean_key}")

        if clean_key.startswith(f"{prefix}/"):
            keys.append(clean_key.replace(f"{prefix}/", "", 1))

    result = []
    seen = set()

    for key in keys:
        key = key.lstrip("/")
        if key and key not in seen:
            result.append(key)
            seen.add(key)

    return result


def load_s3_audio(s3_key: str, bucket: str = S3_BUCKET) -> np.ndarray:
    s3 = get_s3_client()
    candidates = _candidate_s3_keys(s3_key)

    last_error = None

    for candidate_key in candidates:
        try:
            logger.debug("Trying S3 audio bucket=%s key=%s", bucket, candidate_key)

            with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
                s3.download_fileobj(bucket, candidate_key, tmp)
                tmp.flush()

                audio, _ = librosa.load(tmp.name, sr=SR, mono=True)

            logger.debug(
                "Loaded S3 audio bucket=%s key=%s samples=%s sr=%s",
                bucket,
                candidate_key,
                len(audio),
                SR,
            )

            return audio.astype(np.float32)

        except ClientError as exc:
            last_error = exc
            code = exc.response.get("Error", {}).get("Code")
            if code not in {"404", "NoSuchKey", "NotFound"}:
                logger.warning(
                    "S3 audio load failed bucket=%s key=%s error=%s",
                    bucket,
                    candidate_key,
                    exc,
                )

        except Exception as exc:
            last_error = exc
            logger.warning(
                "Audio decode failed bucket=%s key=%s error=%s",
                bucket,
                candidate_key,
                exc,
            )

    raise FileNotFoundError(
        f"Could not load S3 audio. bucket={bucket}, original_key={s3_key}, "
        f"tried={candidates}, last_error={last_error}"
    )


def load_prepared_audio(path: str) -> np.ndarray:
    logger.debug("Loading prepared audio path=%s", path)

    if path.endswith(".npy"):
        audio = np.load(path).astype(np.float32)
        logger.debug("Loaded prepared npy path=%s samples=%s", path, len(audio))
        return audio

    audio, _ = librosa.load(path, sr=SR, mono=True)
    logger.debug("Loaded prepared audio path=%s samples=%s sr=%s", path, len(audio), SR)
    return audio.astype(np.float32)


def normalize_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio)) + 1e-8)
    return (audio / max_abs * peak).astype(np.float32)


def cut_random_segment(audio: np.ndarray, duration_sec: float) -> np.ndarray:
    target_len = int(duration_sec * SR)

    if len(audio) <= target_len:
        out = np.zeros(target_len, dtype=np.float32)
        out[: len(audio)] = audio.astype(np.float32)
        return out

    start = random.randint(0, len(audio) - target_len)
    return audio[start:start + target_len].astype(np.float32)


def add_gaussian_noise(
    audio: np.ndarray,
    snr_db_min: float,
    snr_db_max: float,
) -> np.ndarray:
    snr_db = random.uniform(snr_db_min, snr_db_max)

    signal_power = np.mean(audio ** 2) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    noise = np.random.randn(len(audio)).astype(np.float32)
    noise = noise / (np.std(noise) + 1e-8)
    noise = noise * np.sqrt(noise_power)

    return (audio + noise).astype(np.float32)


def random_gain(audio: np.ndarray, min_db: float = -12.0, max_db: float = 8.0) -> np.ndarray:
    gain_db = random.uniform(min_db, max_db)
    gain = 10 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32)


def random_clipping(audio: np.ndarray, min_clip: float = 0.55, max_clip: float = 0.75) -> np.ndarray:
    clip_val = random.uniform(min_clip, max_clip)
    return np.clip(audio, -clip_val, clip_val).astype(np.float32)


def random_pitch_shift(audio: np.ndarray) -> np.ndarray:
    n_steps = random.uniform(-1.0, 1.0)
    return librosa.effects.pitch_shift(audio, sr=SR, n_steps=n_steps).astype(np.float32)


def random_time_stretch(audio: np.ndarray) -> np.ndarray:
    rate = random.uniform(0.95, 1.05)
    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


def simple_reverb(audio: np.ndarray) -> np.ndarray:
    delay_ms = random.choice([20, 35, 50, 70])
    decay = random.uniform(0.15, 0.45)
    delay = int(SR * delay_ms / 1000.0)

    impulse = np.zeros(delay * 3 + 1, dtype=np.float32)
    impulse[0] = 1.0
    impulse[delay] = decay
    impulse[min(delay * 2, len(impulse) - 1)] = decay * 0.5
    impulse[min(delay * 3, len(impulse) - 1)] = decay * 0.25

    return np.convolve(audio, impulse, mode="full")[: len(audio)].astype(np.float32)


def _bandpass_fft(audio: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    n = len(audio)
    if n <= 1:
        return audio.astype(np.float32)

    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return np.fft.irfft(spectrum * mask, n=n).astype(np.float32)


def add_colored_noise(audio: np.ndarray, snr_db_min: float, snr_db_max: float, color: str = "pink") -> np.ndarray:
    snr_db = random.uniform(snr_db_min, snr_db_max)
    n = len(audio)
    if n <= 1:
        return audio.astype(np.float32)

    white = np.random.randn(n).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0

    if color == "pink":
        scale = 1.0 / np.sqrt(freqs)
    elif color == "brown":
        scale = 1.0 / freqs
    else:
        scale = np.ones_like(freqs)

    colored = np.fft.irfft(spectrum * scale, n=n).astype(np.float32)
    colored = colored / (np.std(colored) + 1e-8)

    signal_power = np.mean(audio ** 2) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    return (audio + colored * np.sqrt(noise_power)).astype(np.float32)


def random_dynamic_compression(audio: np.ndarray) -> np.ndarray:
    threshold = random.uniform(0.12, 0.45)
    ratio = random.uniform(2.0, 8.0)
    x = audio.astype(np.float32)
    sign = np.sign(x)
    mag = np.abs(x)
    over = mag > threshold
    mag[over] = threshold + (mag[over] - threshold) / ratio
    return (sign * mag).astype(np.float32)


def random_phone_filter(audio: np.ndarray, hard: bool = False) -> np.ndarray:
    low = random.uniform(90.0, 260.0) if hard else random.uniform(60.0, 180.0)
    high = random.uniform(4200.0, 7200.0) if hard else random.uniform(5500.0, 7800.0)
    return _bandpass_fft(audio.astype(np.float32), low_hz=low, high_hz=high)


def random_codec_degrade(audio: np.ndarray, hard: bool = False) -> np.ndarray:
    target_sr = random.choice([6000, 8000, 11025, 12000] if hard else [11025, 12000, 14000])
    try:
        down = librosa.resample(audio, orig_sr=SR, target_sr=target_sr)
        up = librosa.resample(down, orig_sr=target_sr, target_sr=SR)
        if len(up) < len(audio):
            up = np.pad(up, (0, len(audio) - len(up)))
        return up[: len(audio)].astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


def random_soft_saturation(audio: np.ndarray) -> np.ndarray:
    drive = random.uniform(1.1, 2.5)
    return (np.tanh(audio.astype(np.float32) * drive) / np.tanh(drive)).astype(np.float32)


def augment_phone(audio: np.ndarray, level: str) -> np.ndarray:
    out = audio.astype(np.float32).copy()
    hard = level == "phone_hard"

    if random.random() < (0.90 if hard else 0.70):
        out = random_gain(out, min_db=-16.0 if hard else -10.0, max_db=9.0 if hard else 6.0)
    if random.random() < (0.75 if hard else 0.55):
        out = random_phone_filter(out, hard=hard)
    if random.random() < (0.60 if hard else 0.35):
        out = random_dynamic_compression(out)
    if random.random() < (0.55 if hard else 0.35):
        out = simple_reverb(out)
    if random.random() < (0.75 if hard else 0.55):
        out = add_colored_noise(
            out,
            snr_db_min=0.0 if hard else 8.0,
            snr_db_max=16.0 if hard else 24.0,
            color=random.choice(["pink", "brown", "white"] if hard else ["pink", "white"]),
        )
    if random.random() < (0.30 if hard else 0.15):
        out = random_soft_saturation(out)
    if random.random() < (0.20 if hard else 0.10):
        out = random_codec_degrade(out, hard=hard)

    return normalize_peak(out, peak=random.uniform(0.86, 0.98))


def augment_phone_noisy(audio: np.ndarray) -> np.ndarray:
    r = random.random()
    if r < 0.65:
        return augment_phone(audio, "phone_mild")
    if r < 0.92:
        return augment_phone(audio, "phone_medium")
    return augment_phone(audio, "phone_hard")


def apply_corruption(audio: np.ndarray, corruption: str) -> np.ndarray:
    out = audio.astype(np.float32).copy()

    if corruption == "clean":
        return normalize_peak(out)

    if corruption in {"noisy", "noise_snr20"}:
        out = add_gaussian_noise(out, 18.0, 22.0)
    elif corruption == "noise_snr10":
        out = add_gaussian_noise(out, 9.0, 11.0)
    elif corruption == "noise_snr5":
        out = add_gaussian_noise(out, 4.0, 6.0)
    elif corruption == "strong_noisy":
        out = random_gain(out, min_db=-14.0, max_db=10.0)
        out = add_gaussian_noise(out, snr_db_min=-2.0, snr_db_max=18.0)
        if random.random() < 0.45:
            out = random_clipping(out, min_clip=0.45, max_clip=0.90)
        if random.random() < 0.45:
            out = simple_reverb(out)
    elif corruption in {"phone_mild", "phone_medium", "phone_hard"}:
        out = augment_phone(out, corruption)
    elif corruption == "phone_noisy":
        out = augment_phone_noisy(out)
    elif corruption == "reverb":
        out = simple_reverb(out)
    elif corruption == "clipping":
        out = random_clipping(out)
    elif corruption == "gain":
        out = random_gain(out)
    elif corruption == "pitch_shift":
        out = random_pitch_shift(out)
    elif corruption == "time_stretch":
        out = random_time_stretch(out)
    elif corruption == "mixed_noise_reverb":
        out = add_gaussian_noise(out, 8.0, 14.0)
        out = simple_reverb(out)
    else:
        raise ValueError(f"Unsupported corruption: {corruption}")

    return normalize_peak(out)
