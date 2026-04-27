
import numpy as np
from scipy.ndimage import maximum_filter
from config.config import PEAK_NEIGHBORHOOD_SIZE, AMP_MIN_DB

def find_peaks(S_db):
    neighborhood = maximum_filter(S_db, size=PEAK_NEIGHBORHOOD_SIZE)
    mask = (S_db == neighborhood) & (S_db > AMP_MIN_DB)

    freq_idx, time_idx = np.where(mask)

    candidates = list(zip(time_idx, freq_idx, S_db[freq_idx, time_idx]))
    candidates.sort(key=lambda x: (x[0], -x[2]))

    max_peaks_per_frame = 8
    result = []
    counts = {}

    for t, f, amp in candidates:
        count = counts.get(t, 0)
        if count >= max_peaks_per_frame:
            continue
        result.append((int(t), int(f)))
        counts[t] = count + 1

    return result