import numpy as np


def normalize(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak

    return audio