import librosa
import numpy as np

def spectrogram(audio, sr):
    S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db