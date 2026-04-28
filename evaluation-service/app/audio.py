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


def apply_corruption(audio: np.ndarray, corruption: str) -> np.ndarray:
    out = audio.astype(np.float32).copy()

    if corruption == "clean":
        return normalize_peak(out)

    if corruption == "noise_snr20":
        out = add_gaussian_noise(out, 18.0, 22.0)
    elif corruption == "noise_snr10":
        out = add_gaussian_noise(out, 9.0, 11.0)
    elif corruption == "noise_snr5":
        out = add_gaussian_noise(out, 4.0, 6.0)
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