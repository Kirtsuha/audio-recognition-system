import numpy as np
import librosa
from config import *

def add_noise(audio, noise_level=NOISE_LEVEL):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

def pitch_shift(audio, sr):
    return librosa.effects.pitch_shift(audio, sr, n_steps=PITCH_SHIFT_N_STEPS)

def time_stretch(audio):
    return librosa.effects.time_stretch(audio, rate=TIME_STRETCH_RATE)