import librosa

from audio.loader import load_audio
from audio.preprocess import normalize
from audio.spectrogram import spectrogram

import matplotlib.pyplot as plt
import numpy as np

from config.config import SAMPLE_RATE, HOP_LENGTH, N_FFT
from fingerprint.peaks import find_peaks

AUDIO_PATH = "../data/Never_Gonna.wav"

y, sr = load_audio(AUDIO_PATH, sr=SAMPLE_RATE, duration=15.0)
S_db1 = spectrogram(y, sr)

peaks = find_peaks(S_db1)

times = librosa.frames_to_time([t for t, f in peaks], sr=sr, hop_length=HOP_LENGTH)
freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)[[f for t, f in peaks]]

plt.figure()
times_spec = np.arange(S_db1.shape[1]) * HOP_LENGTH / sr
freqs_spec = np.fft.rfftfreq(N_FFT, 1/sr)

plt.imshow(S_db1, origin='lower', aspect='auto',
           extent=[times_spec[0], times_spec[-1], freqs_spec[0], freqs_spec[-1]],
           cmap='magma')

plt.scatter(times, freqs, s=15, c='cyan', marker='o')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram with detected peaks")
plt.colorbar(format="%+2.0f dB")
plt.show()