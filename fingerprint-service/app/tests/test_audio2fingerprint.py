import numpy as np
from audio.preprocess import normalize
from fingerprint.peaks import find_peaks
from fingerprint.hashgen import generate_hashes

def test_normalize():
    arr = np.array([0, 2, 4, 8])
    norm = normalize(arr)
    assert np.max(norm) == 1.0

def test_find_peaks_simple():
    S_db = np.array([[0, -50, 0],
                     [0, -30, 0],
                     [0, 0, 0]])
    peaks = find_peaks(S_db)
    assert isinstance(peaks, list)

def test_generate_hashes_ordered_peaks():
    peaks = [(0, 10), (1, 20), (2, 30)]
    hashes = generate_hashes(peaks)
    assert all(isinstance(h[0], int) for h in hashes)
    assert all(isinstance(h[1], int) for h in hashes)