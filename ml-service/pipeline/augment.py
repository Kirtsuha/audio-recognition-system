import random
import numpy as np
import librosa

from pipeline.config import SR, FAST_AUGMENT


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


def augment_audio(audio: np.ndarray, fast_mode: bool = FAST_AUGMENT) -> np.ndarray:


    out = audio.astype(np.float32).copy()

    if random.random() < 0.75:
        out = random_gain(out)

    if random.random() < 0.70:
        out = add_gaussian_noise(out)

    if random.random() < 0.30:
        out = random_clipping(out)

    if random.random() < 0.25:
        out = simple_reverb(out)

    if not fast_mode:
        if random.random() < 0.20:
            out = random_pitch_shift(out)

        if random.random() < 0.20:
            out = random_time_stretch(out)

    out = normalize_peak(out)
    return out.astype(np.float32)

def augment_audio_strong_noisy(audio: np.ndarray) -> np.ndarray:
    out = audio.astype(np.float32).copy()


    if random.random() < 0.95:
        out = random_gain(out, min_db=-14.0, max_db=10.0)


    if random.random() < 0.95:
        out = add_gaussian_noise(out, snr_db_min=-2.0, snr_db_max=18.0)


    if random.random() < 0.45:
        out = random_clipping(out, min_clip=0.45, max_clip=0.90)


    if random.random() < 0.45:
        out = simple_reverb(out)


    if random.random() < 0.25:
        out = add_gaussian_noise(out, snr_db_min=5.0, snr_db_max=20.0)

    out = normalize_peak(out)
    return out.astype(np.float32)

def _bandpass_fft(audio: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    n = len(audio)
    if n <= 1:
        return audio.astype(np.float32)

    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)

    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum = spectrum * mask

    out = np.fft.irfft(spectrum, n=n).astype(np.float32)
    return out


def add_colored_noise(
    audio: np.ndarray,
    snr_db_min: float = 0.0,
    snr_db_max: float = 18.0,
    color: str = "pink",
) -> np.ndarray:
    snr_db = random.uniform(snr_db_min, snr_db_max)

    n = len(audio)
    if n <= 1:
        return audio.astype(np.float32)

    white = np.random.randn(n).astype(np.float32)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)

    if len(freqs) > 1:
        freqs[0] = freqs[1]
    else:
        freqs[0] = 1.0

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
    colored = colored * np.sqrt(noise_power)

    return (audio + colored).astype(np.float32)


def random_phone_filter(audio: np.ndarray) -> np.ndarray:
    low = random.uniform(120.0, 350.0)
    high = random.uniform(3200.0, 7200.0)

    out = audio.astype(np.float32)

    if random.random() < 0.70:
        try:
            out = librosa.effects.preemphasis(out, coef=random.uniform(0.85, 0.98))
        except Exception:
            pass

    return _bandpass_fft(out, low_hz=low, high_hz=high)


def random_dynamic_compression(audio: np.ndarray) -> np.ndarray:
    threshold = random.uniform(0.12, 0.45)
    ratio = random.uniform(2.0, 8.0)

    x = audio.astype(np.float32)
    sign = np.sign(x)
    mag = np.abs(x)

    over = mag > threshold
    mag_compressed = mag.copy()
    mag_compressed[over] = threshold + (mag[over] - threshold) / ratio

    return (sign * mag_compressed).astype(np.float32)


def random_time_dropout(audio: np.ndarray) -> np.ndarray:
    out = audio.astype(np.float32).copy()

    drops = random.randint(1, 4)
    for _ in range(drops):
        dur_ms = random.uniform(20.0, 120.0)
        dur = int(SR * dur_ms / 1000.0)

        if dur <= 0 or dur >= len(out):
            continue

        start = random.randint(0, len(out) - dur)
        scale = random.uniform(0.0, 0.25)
        out[start:start + dur] *= scale

    return out


def random_codec_degrade(audio: np.ndarray) -> np.ndarray:
    target_sr = random.choice([6000, 8000, 11025, 12000])

    try:
        down = librosa.resample(audio, orig_sr=SR, target_sr=target_sr)
        up = librosa.resample(down, orig_sr=target_sr, target_sr=SR)

        if len(up) < len(audio):
            up = np.pad(up, (0, len(audio) - len(up)))
        else:
            up = up[: len(audio)]

        return up.astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


def random_room_reverb(audio: np.ndarray) -> np.ndarray:
    delay_ms_1 = random.uniform(15.0, 45.0)
    delay_ms_2 = random.uniform(50.0, 120.0)
    delay_ms_3 = random.uniform(120.0, 220.0)

    d1 = int(SR * delay_ms_1 / 1000.0)
    d2 = int(SR * delay_ms_2 / 1000.0)
    d3 = int(SR * delay_ms_3 / 1000.0)

    impulse_len = max(d1, d2, d3) + 1
    impulse = np.zeros(impulse_len, dtype=np.float32)

    impulse[0] = 1.0
    impulse[d1] += random.uniform(0.10, 0.40)
    impulse[d2] += random.uniform(0.05, 0.25)
    impulse[d3] += random.uniform(0.02, 0.15)

    reverbed = np.convolve(audio, impulse, mode="full")[: len(audio)]
    return reverbed.astype(np.float32)

def random_soft_saturation(audio: np.ndarray) -> np.ndarray:


    drive = random.uniform(1.1, 2.5)
    out = np.tanh(audio.astype(np.float32) * drive) / np.tanh(drive)
    return out.astype(np.float32)


def random_phone_filter_mild(audio: np.ndarray) -> np.ndarray:


    low = random.uniform(60.0, 180.0)
    high = random.uniform(5500.0, 7800.0)

    out = audio.astype(np.float32)

    if random.random() < 0.35:
        try:
            out = librosa.effects.preemphasis(out, coef=random.uniform(0.75, 0.92))
        except Exception:
            pass

    return _bandpass_fft(out, low_hz=low, high_hz=high)


def random_phone_filter_medium(audio: np.ndarray) -> np.ndarray:


    low = random.uniform(90.0, 260.0)
    high = random.uniform(4200.0, 7200.0)

    out = audio.astype(np.float32)

    if random.random() < 0.50:
        try:
            out = librosa.effects.preemphasis(out, coef=random.uniform(0.80, 0.95))
        except Exception:
            pass

    return _bandpass_fft(out, low_hz=low, high_hz=high)


def random_room_reverb_mild(audio: np.ndarray) -> np.ndarray:


    delay_ms_1 = random.uniform(12.0, 35.0)
    delay_ms_2 = random.uniform(40.0, 90.0)

    d1 = int(SR * delay_ms_1 / 1000.0)
    d2 = int(SR * delay_ms_2 / 1000.0)

    impulse_len = max(d1, d2) + 1
    impulse = np.zeros(impulse_len, dtype=np.float32)

    impulse[0] = 1.0
    impulse[d1] += random.uniform(0.04, 0.18)
    impulse[d2] += random.uniform(0.02, 0.10)

    reverbed = np.convolve(audio, impulse, mode="full")[: len(audio)]
    return reverbed.astype(np.float32)


def random_codec_degrade_mild(audio: np.ndarray) -> np.ndarray:


    target_sr = random.choice([11025, 12000, 14000])

    try:
        down = librosa.resample(audio, orig_sr=SR, target_sr=target_sr)
        up = librosa.resample(down, orig_sr=target_sr, target_sr=SR)

        if len(up) < len(audio):
            up = np.pad(up, (0, len(audio) - len(up)))
        else:
            up = up[: len(audio)]

        return up.astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


def augment_audio_phone_mild(audio: np.ndarray) -> np.ndarray:


    out = audio.astype(np.float32).copy()

    if random.random() < 0.70:
        out = random_gain(out, min_db=-10.0, max_db=6.0)

    if random.random() < 0.55:
        out = random_phone_filter_mild(out)

    if random.random() < 0.35:
        out = random_dynamic_compression(out)

    if random.random() < 0.35:
        out = random_room_reverb_mild(out)

    if random.random() < 0.55:
        out = add_colored_noise(
            out,
            snr_db_min=8.0,
            snr_db_max=24.0,
            color=random.choice(["pink", "white"]),
        )

    if random.random() < 0.15:
        out = random_soft_saturation(out)

    if random.random() < 0.10:
        out = random_codec_degrade_mild(out)


    out = normalize_peak(out, peak=random.uniform(0.88, 0.98))
    return out.astype(np.float32)


def augment_audio_phone_medium(audio: np.ndarray) -> np.ndarray:


    out = audio.astype(np.float32).copy()

    if random.random() < 0.85:
        out = random_gain(out, min_db=-14.0, max_db=8.0)

    if random.random() < 0.70:
        out = random_phone_filter_medium(out)

    if random.random() < 0.50:
        out = random_dynamic_compression(out)

    if random.random() < 0.50:
        out = random_room_reverb_mild(out)

    if random.random() < 0.65:
        out = add_colored_noise(
            out,
            snr_db_min=4.0,
            snr_db_max=20.0,
            color=random.choice(["pink", "brown", "white"]),
        )

    if random.random() < 0.25:
        out = random_soft_saturation(out)

    if random.random() < 0.18:
        out = random_codec_degrade_mild(out)

    if random.random() < 0.08:
        out = random_clipping(out, min_clip=0.65, max_clip=0.95)

    out = normalize_peak(out, peak=random.uniform(0.88, 0.98))
    return out.astype(np.float32)


def augment_audio_phone_hard(audio: np.ndarray) -> np.ndarray:


    out = audio.astype(np.float32).copy()

    if random.random() < 0.90:
        out = random_gain(out, min_db=-16.0, max_db=9.0)

    if random.random() < 0.75:
        out = random_phone_filter_medium(out)

    if random.random() < 0.60:
        out = random_dynamic_compression(out)

    if random.random() < 0.55:
        out = random_room_reverb(out)

    if random.random() < 0.75:
        out = add_colored_noise(
            out,
            snr_db_min=0.0,
            snr_db_max=16.0,
            color=random.choice(["pink", "brown", "white"]),
        )

    if random.random() < 0.30:
        out = random_soft_saturation(out)

    if random.random() < 0.20:
        out = random_codec_degrade(out)

    if random.random() < 0.10:
        out = random_time_dropout(out)

    if random.random() < 0.12:
        out = random_clipping(out, min_clip=0.55, max_clip=0.90)

    out = normalize_peak(out, peak=random.uniform(0.86, 0.98))
    return out.astype(np.float32)


def augment_audio_phone_noisy(audio: np.ndarray) -> np.ndarray:


    r = random.random()

    if r < 0.65:
        return augment_audio_phone_mild(audio)

    if r < 0.92:
        return augment_audio_phone_medium(audio)

    return augment_audio_phone_hard(audio)