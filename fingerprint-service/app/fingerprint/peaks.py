
import numpy as np
from scipy.ndimage import maximum_filter
from config.config import PEAK_NEIGHBORHOOD_SIZE, AMP_MIN_DB

def find_peaks(S_db):
    neighborhood = maximum_filter(S_db, size=PEAK_NEIGHBORHOOD_SIZE)
    peaks = (S_db == neighborhood) & (S_db > AMP_MIN_DB)

    print(f"=== DEBUG find_peaks ===")
    print(f"S_db shape: {S_db.shape}")
    print(f"S_db ndim: {S_db.ndim}")
    print(f"peaks shape: {peaks.shape}")
    print(f"peaks ndim: {peaks.ndim}")
    print(f"np.where(peaks) result length: {len(np.where(peaks))}")

    freq_idx, time_idx = np.where(peaks)
    return list(zip(time_idx, freq_idx))