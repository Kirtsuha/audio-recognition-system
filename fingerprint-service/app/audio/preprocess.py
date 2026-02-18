import numpy as np

def normalize(audio):
    if np.max(audio) > 1:
        audio = audio/np.max(audio)
    return audio