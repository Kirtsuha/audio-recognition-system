import librosa


def load_audio(path, sr=44100, duration=15):
    y, sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    return y, sr