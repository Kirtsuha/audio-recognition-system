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
FIXED_EVAL_DIR = Path(os.getenv("FIXED_EVAL_DIR", str(EVAL_DIR / "fixed_sets")))

# Prepared dataset
PREPARED_DATA_DIR = Path(os.getenv("PREPARED_DATA_DIR", "/app/prepared-data"))
PREPARED_DATASET_DIR = Path(os.getenv("PREPARED_DATASET_DIR", str(PREPARED_DATA_DIR / "dataset_v1")))
PREPARED_TRACKS_DIR = Path(os.getenv("PREPARED_TRACKS_DIR", str(PREPARED_DATASET_DIR / "tracks")))
PREPARED_MANIFEST_PATH = Path(os.getenv("PREPARED_MANIFEST_PATH", str(PREPARED_DATASET_DIR / "manifest.jsonl")))
INVALID_KEYS_PATH = Path(os.getenv("INVALID_KEYS_PATH", str(PREPARED_DATASET_DIR / "invalid_keys.json")))

# Audio
SR = int(os.getenv("SR", "16000"))
SEGMENT_SECONDS = float(os.getenv("SEGMENT_SECONDS", "15.0"))
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
INDEX_WINDOWS_PER_SONG = int(os.getenv("INDEX_WINDOWS_PER_SONG", "96"))
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "ivf_flat")  # flat | ivf_flat
FAISS_NLIST = int(os.getenv("FAISS_NLIST", "4096"))
FAISS_NPROBE = int(os.getenv("FAISS_NPROBE", "32"))
FAISS_INDEX_META_PATH = Path(
    os.getenv("FAISS_INDEX_META_PATH", str(ACTIVE_ARTIFACTS_DIR / "faiss_meta.json"))
)

AGGREGATION_STRATEGY = os.getenv("AGGREGATION_STRATEGY", "max") # potentially change to max

# Acceptance thresholds
# MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.58"))
# MIN_MARGIN = float(os.getenv("MIN_MARGIN", "0.10"))
# MIN_SUPPORTED_WINDOWS = int(os.getenv("MIN_SUPPORTED_WINDOWS", "2"))
MIN_CONFIDENCE=0.55
MIN_MARGIN=0.0
MIN_SUPPORTED_WINDOWS=1

# Train
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "24"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
LR = float(os.getenv("LR", "1e-3"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
MARGIN = float(os.getenv("MARGIN", "0.30"))
TRAIN_NEGATIVE_RETRIES = int(os.getenv("TRAIN_NEGATIVE_RETRIES", "10"))
TRAIN_LOSS_TYPE = os.getenv("TRAIN_LOSS_TYPE", "infonce")  # triplet | infonce
INFONCE_TEMPERATURE = float(os.getenv("INFONCE_TEMPERATURE", "0.05"))
SAVE_EPOCH_CHECKPOINTS = os.getenv("SAVE_EPOCH_CHECKPOINTS", "true").lower() == "true"

# Experiment defaults
EXPERIMENT_RANDOM_SEED = int(os.getenv("EXPERIMENT_RANDOM_SEED", "42"))

# Performance / augmentation
FAST_AUGMENT = os.getenv("FAST_AUGMENT", "true").lower() == "true"
CACHE_PREPARED_IN_MEMORY = os.getenv("CACHE_PREPARED_IN_MEMORY", "false").lower() == "true"
CACHE_PREPARED_MAX_TRACKS = int(os.getenv("CACHE_PREPARED_MAX_TRACKS", "0"))
TORCHAUDIO_MEL_DEVICE = os.getenv("TORCHAUDIO_MEL_DEVICE", "cpu")

#Production pipeline defaults
FULL_RETRAIN_NEW_TRACK_RATIO = float(os.getenv("FULL_RETRAIN_NEW_TRACK_RATIO", "0.25"))
FULL_RETRAIN_NEW_TRACK_MIN = int(os.getenv("FULL_RETRAIN_NEW_TRACK_MIN", "2000"))

KEEP_LAST_FULL_RUNS = int(os.getenv("KEEP_LAST_FULL_RUNS", "2"))
KEEP_LAST_EXPERIMENT_RUNS = int(os.getenv("KEEP_LAST_EXPERIMENT_RUNS", "5"))

RUN_METADATA_FILENAME = os.getenv("RUN_METADATA_FILENAME", "run_metadata.json")

INFONCE_AUG_LIGHT_PROB = float(os.getenv("INFONCE_AUG_LIGHT_PROB", "0.15"))
INFONCE_AUG_STRONG_PROB = float(os.getenv("INFONCE_AUG_STRONG_PROB", "0.40"))
INFONCE_AUG_PHONE_MILD_PROB = float(os.getenv("INFONCE_AUG_PHONE_MILD_PROB", "0.30"))
INFONCE_AUG_PHONE_MEDIUM_PROB = float(os.getenv("INFONCE_AUG_PHONE_MEDIUM_PROB", "0.12"))
INFONCE_AUG_PHONE_HARD_PROB = float(os.getenv("INFONCE_AUG_PHONE_HARD_PROB", "0.03"))

RERANKER_REF_EMBEDDINGS_PATH = Path(
    os.getenv("RERANKER_REF_EMBEDDINGS_PATH", str(ACTIVE_ARTIFACTS_DIR / "reranker_ref_embeddings.npy"))
)
RERANKER_REF_META_PATH = Path(
    os.getenv("RERANKER_REF_META_PATH", str(ACTIVE_ARTIFACTS_DIR / "reranker_ref_meta.jsonl"))
)
RERANKER_REF_CONFIG_PATH = Path(
    os.getenv("RERANKER_REF_CONFIG_PATH", str(ACTIVE_ARTIFACTS_DIR / "reranker_ref_config.json"))
)

RERANKER_REF_WINDOW_SECONDS = float(os.getenv("RERANKER_REF_WINDOW_SECONDS", str(SEGMENT_SECONDS)))
RERANKER_REF_HOP_SECONDS = float(os.getenv("RERANKER_REF_HOP_SECONDS", "2.0"))
RERANKER_REF_BATCH_SIZE = int(os.getenv("RERANKER_REF_BATCH_SIZE", "64"))