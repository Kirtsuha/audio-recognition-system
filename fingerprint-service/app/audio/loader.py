import librosa
import soundfile as sf

from config.config import SAMPLE_RATE


def load_audio(path, sr=SAMPLE_RATE, duration=15):
    y, file_sr = sf.read(path)
    y = y[:sr * duration]
    if file_sr != sr:
        y = librosa.resample(y=y, orig_sr=file_sr, target_sr=sr)
    return y, sr