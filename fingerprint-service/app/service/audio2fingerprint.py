from audio.loader import load_audio
from audio.preprocess import normalize
from audio.spectrogram import spectrogram
from fingerprint.peaks import find_peaks
from fingerprint.hashgen import generate_hashes
from config.config import SAMPLE_RATE, RECOGNITION_DURATION_SEC


def fingerprint_audio(path: str, duration: float | None = None):
    audio, sr = load_audio(
        path,
        sr=SAMPLE_RATE,
        duration=duration if duration is not None else RECOGNITION_DURATION_SEC,
    )

    if audio is None or sr is None:
        return []

    audio = normalize(audio)

    if len(audio) == 0:
        return []

    s_db = spectrogram(audio, sr)
    peaks = find_peaks(s_db)
    return generate_hashes(peaks)