import librosa

from pathlib import Path
from audio.loader import load_audio
from audio.preprocess import normalize
from audio.spectrogram import spectrogram

import matplotlib.pyplot as plt

AUDIO_PATH = "../data/Never_Gonna.wav"

y, sr = load_audio(AUDIO_PATH, sr=44100, duration=5.0)
S_db1 = spectrogram(y, sr)
S_db2 = spectrogram(normalize(y), sr)

plt.figure()

plt.subplot(2,1,1)
librosa.display.specshow(S_db1, x_axis='time', y_axis='linear')
plt.title("Spectrogram BEFORE normalization")
plt.colorbar(format="%+2.0f dB")

plt.subplot(2,1,2)
librosa.display.specshow(S_db2, x_axis='time', y_axis='linear')
plt.title("Spectrogram AFTER normalization")
plt.colorbar(format="%+2.0f dB")

plt.show()