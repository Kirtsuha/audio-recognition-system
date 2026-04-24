from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Online active artifacts
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/app/artifacts"))
ACTIVE_ARTIFACTS_DIR = Path(os.getenv("ACTIVE_ARTIFACTS_DIR", str(ARTIFACTS_DIR / "active")))
RUNS_DIR = Path(os.getenv("RUNS_DIR", str(ARTIFACTS_DIR / "runs")))
JOB_STATUS_PATH = Path(os.getenv("JOB_STATUS_PATH", str(ARTIFACTS_DIR / "job_status.json")))

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ACTIVE_ARTIFACTS_DIR / "model.pt")))
EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", str(ACTIVE_ARTIFACTS_DIR / "embeddings.npy")))
SONG_IDS_PATH = Path(os.getenv("SONG_IDS_PATH", str(ACTIVE_ARTIFACTS_DIR / "song_ids.npy")))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(ACTIVE_ARTIFACTS_DIR / "faiss.index")))
SONG_MANIFEST_PATH = Path(os.getenv("SONG_MANIFEST_PATH", str(ACTIVE_ARTIFACTS_DIR / "song_manifest.json")))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(ACTIVE_ARTIFACTS_DIR / "metrics.json")))

# Evaluation artifacts
EVAL_DIR = Path(os.getenv("EVAL_DIR", str(ARTIFACTS_DIR / "eval")))
EVAL_QUERIES_PATH = Path(os.getenv("EVAL_QUERIES_PATH", str(EVAL_DIR / "eval_queries.jsonl")))
EVAL_RESULTS_PATH = Path(os.getenv("EVAL_RESULTS_PATH", str(EVAL_DIR / "eval_results.jsonl")))
EVAL_METRICS_PATH = Path(os.getenv("EVAL_METRICS_PATH", str(EVAL_DIR / "eval_metrics.json")))
THRESHOLD_TUNING_PATH = Path(os.getenv("THRESHOLD_TUNING_PATH", str(EVAL_DIR / "threshold_tuning.json")))

# Prepared dataset
PREPARED_DATA_DIR = Path(os.getenv("PREPARED_DATA_DIR", "/app/prepared-data"))
PREPARED_DATASET_DIR = Path(os.getenv("PREPARED_DATASET_DIR", str(PREPARED_DATA_DIR / "dataset_v1")))
PREPARED_TRACKS_DIR = Path(os.getenv("PREPARED_TRACKS_DIR", str(PREPARED_DATASET_DIR / "tracks")))
PREPARED_MANIFEST_PATH = Path(os.getenv("PREPARED_MANIFEST_PATH", str(PREPARED_DATASET_DIR / "manifest.jsonl")))
INVALID_KEYS_PATH = Path(os.getenv("INVALID_KEYS_PATH", str(PREPARED_DATASET_DIR / "invalid_keys.json")))

# Audio
SR = int(os.getenv("SR", "16000"))
SEGMENT_SECONDS = float(os.getenv("SEGMENT_SECONDS", "12.0"))
WINDOW_HOP_SECONDS = float(os.getenv("WINDOW_HOP_SECONDS", "1.0"))
MAX_QUERY_WINDOWS = int(os.getenv("MAX_QUERY_WINDOWS", "24"))

# Mel
N_MELS = int(os.getenv("N_MELS", "96"))
N_FFT = int(os.getenv("N_FFT", "1024"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "256"))
WIN_LENGTH = int(os.getenv("WIN_LENGTH", "1024"))
FMIN = float(os.getenv("FMIN", "20"))
FMAX = float(os.getenv("FMAX", str(SR // 2)))

# Embeddings / Index
EMB_DIM = int(os.getenv("EMB_DIM", "128"))
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "15"))
INDEX_WINDOWS_PER_SONG = int(os.getenv("INDEX_WINDOWS_PER_SONG", "24"))

# Acceptance thresholds
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.58"))
MIN_MARGIN = float(os.getenv("MIN_MARGIN", "0.10"))
MIN_SUPPORTED_WINDOWS = int(os.getenv("MIN_SUPPORTED_WINDOWS", "2"))

# Train
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "24"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))
LR = float(os.getenv("LR", "1e-3"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
MARGIN = float(os.getenv("MARGIN", "0.30"))
TRAIN_NEGATIVE_RETRIES = int(os.getenv("TRAIN_NEGATIVE_RETRIES", "10"))

# Experiment defaults
EXPERIMENT_RANDOM_SEED = int(os.getenv("EXPERIMENT_RANDOM_SEED", "42"))

# Performance / augmentation
FAST_AUGMENT = os.getenv("FAST_AUGMENT", "true").lower() == "true"
CACHE_PREPARED_IN_MEMORY = os.getenv("CACHE_PREPARED_IN_MEMORY", "false").lower() == "true"
CACHE_PREPARED_MAX_TRACKS = int(os.getenv("CACHE_PREPARED_MAX_TRACKS", "0"))
TORCHAUDIO_MEL_DEVICE = os.getenv("TORCHAUDIO_MEL_DEVICE", "cpu")