import librosa
import random

from pipeline.config import *


def load_audio(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio

def random_segment(audio):
    max_start = len(audio) - SEGMENT_SECONDS * SR
    start = random.randint(0, max_start)
    end = start + SEGMENT_SECONDS * SR
    return audio[start:end]

def make_training_pair(audio):
    seg1 = random_segment(audio)
    seg2 = random_segment(audio)

    return seg1, seg2