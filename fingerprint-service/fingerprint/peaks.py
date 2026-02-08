
import numpy as np
from scipy.ndimage import maximum_filter
from config.config import PEAK_NEIGHBORHOOD_SIZE, AMP_MIN_DB

def find_peaks(S_db):
    neighborhood = maximum_filter(S_db, size=PEAK_NEIGHBORHOOD_SIZE)
    peaks = (S_db == neighborhood) & (S_db > AMP_MIN_DB)

    freq_idx, time_idx = np.where(peaks)
    return list(zip(time_idx, freq_idx))