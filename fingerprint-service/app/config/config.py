import os

SAMPLE_RATE = int(os.getenv("FP_SAMPLE_RATE", "11025"))

N_FFT = int(os.getenv("FP_N_FFT", "2048"))
HOP_LENGTH = int(os.getenv("FP_HOP_LENGTH", "512"))

PEAK_NEIGHBORHOOD_SIZE = int(os.getenv("FP_PEAK_NEIGHBORHOOD_SIZE", "12"))
AMP_MIN_DB = float(os.getenv("FP_AMP_MIN_DB", "-40"))

FAN_VALUE = int(os.getenv("FP_FAN_VALUE", "8"))
MIN_DELTA_T = int(os.getenv("FP_MIN_DELTA_T", "2"))
MAX_DELTA_T = int(os.getenv("FP_MAX_DELTA_T", "60"))

MAX_PEAKS_PER_FRAME = int(os.getenv("FP_MAX_PEAKS_PER_FRAME", "4"))

MIN_ALIGNED_MATCHES = int(os.getenv("FP_MIN_ALIGNED_MATCHES", "15"))
MIN_QUERY_COVERAGE = float(os.getenv("FP_MIN_QUERY_COVERAGE", "0.002"))
MIN_SCORE_GAP = float(os.getenv("FP_MIN_SCORE_GAP", "1.2"))

_raw_index_duration = os.getenv("FP_INDEX_DURATION_SEC", "")
INDEX_DURATION_SEC = None if _raw_index_duration == "" else float(_raw_index_duration)

RECOGNITION_DURATION_SEC = float(os.getenv("FP_RECOGNITION_DURATION_SEC", "15"))

FREQ_BIN_SIZE = int(os.getenv("FP_FREQ_BIN_SIZE", "2"))
DELTA_T_BIN_SIZE = int(os.getenv("FP_DELTA_T_BIN_SIZE", "2"))
OFFSET_TOLERANCE_FRAMES = int(os.getenv("FP_OFFSET_TOLERANCE_FRAMES", "3"))
OFFSET_BIN = int(os.getenv("FP_OFFSET_BIN", "3"))