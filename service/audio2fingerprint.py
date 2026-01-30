from audio.loader import load_audio
from audio.preprocess import normalize
from audio.spectrogram import spectrogram
from fingerprint.peaks import find_peaks
from fingerprint.hashgen import generate_hashes
from config.config import SAMPLE_RATE

def fingerprint_audio(path: str):
    audio, sr = load_audio(path, sr=SAMPLE_RATE)
    audio = normalize(audio)
    S_db = spectrogram(audio, sr)
    peaks = find_peaks(S_db)
    hashes = generate_hashes(peaks)
    return hashes