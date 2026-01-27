import librosa
import numpy as np

from config.config import N_FFT, HOP_LENGTH

def spectrogram(audio, sr):
    S = np.abs(librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window='hann'
    ))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db