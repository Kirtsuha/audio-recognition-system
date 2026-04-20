from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Audio
SR = int(os.getenv("SR", "16000"))
SEGMENT_SECONDS = float(os.getenv("SEGMENT_SECONDS", "5.0"))
WINDOW_HOP_SECONDS = float(os.getenv("WINDOW_HOP_SECONDS", "2.0"))
MAX_QUERY_WINDOWS = int(os.getenv("MAX_QUERY_WINDOWS", "8"))

# Mel
N_MELS = int(os.getenv("N_MELS", "96"))
N_FFT = int(os.getenv("N_FFT", "1024"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "256"))
WIN_LENGTH = int(os.getenv("WIN_LENGTH", "1024"))
FMIN = float(os.getenv("FMIN", "20"))
FMAX = float(os.getenv("FMAX", str(SR // 2)))

# Embeddings / FAISS
EMB_DIM = int(os.getenv("EMB_DIM", "128"))
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "8"))
INDEX_WINDOWS_PER_SONG = int(os.getenv("INDEX_WINDOWS_PER_SONG", "12"))

# Inference thresholds
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.58"))
MIN_MARGIN = float(os.getenv("MIN_MARGIN", "0.10"))
MIN_SUPPORTED_WINDOWS = int(os.getenv("MIN_SUPPORTED_WINDOWS", "2"))

# Train
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "24"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
LR = float(os.getenv("LR", "1e-3"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
MARGIN = float(os.getenv("MARGIN", "0.30"))

# Paths
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DATA_DIR / "model.pt")))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(DATA_DIR / "faiss.index")))
SONG_IDS_PATH = Path(os.getenv("SONG_IDS_PATH", str(DATA_DIR / "song_ids.npy")))
EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", str(DATA_DIR / "embeddings.npy")))