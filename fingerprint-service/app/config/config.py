import os

SAMPLE_RATE = int(os.getenv("FP_SAMPLE_RATE", "11025"))

N_FFT = int(os.getenv("FP_N_FFT", "2048"))
HOP_LENGTH = int(os.getenv("FP_HOP_LENGTH", "256"))

PEAK_NEIGHBORHOOD_SIZE = int(os.getenv("FP_PEAK_NEIGHBORHOOD_SIZE", "8"))
AMP_MIN_DB = float(os.getenv("FP_AMP_MIN_DB", "-45"))

FAN_VALUE = int(os.getenv("FP_FAN_VALUE", "15"))
MIN_DELTA_T = int(os.getenv("FP_MIN_DELTA_T", "2"))
MAX_DELTA_T = int(os.getenv("FP_MAX_DELTA_T", "100"))

MIN_ALIGNED_MATCHES = int(os.getenv("FP_MIN_ALIGNED_MATCHES", "8"))
MIN_QUERY_COVERAGE = float(os.getenv("FP_MIN_QUERY_COVERAGE", "0.003"))
MIN_SCORE_GAP = float(os.getenv("FP_MIN_SCORE_GAP", "1.05"))

RECOGNITION_DURATION_SEC = float(os.getenv("FP_RECOGNITION_DURATION_SEC", "15"))
MAX_PEAKS_PER_FRAME = int(os.getenv("FP_MAX_PEAKS_PER_FRAME", "8"))