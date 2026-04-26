import librosa
import soundfile as sf
import numpy as np

from config.config import SAMPLE_RATE


def load_audio(path, sr=SAMPLE_RATE, duration=15):
    try:
        y, file_sr = librosa.load(
            path,
            sr=sr,
            mono=True,
            duration=duration,
        )

        # Проверка 1: пустой файл
        if y is None or len(y) == 0:
            print("Empty audio: %s", path)
            return None, None

        # Проверка 3: NaN / Inf
        if not np.isfinite(y).all():
            print("Invalid values in audio: %s", path)
            return None, None

        return y, file_sr

    except Exception as e:
        print("Audio loading failed for %s: %s", path, e)
        return None, None