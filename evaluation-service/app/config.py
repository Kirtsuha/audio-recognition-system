import os
from pathlib import Path

FINGERPRINT_SERVICE_URL = os.getenv(
    "FINGERPRINT_SERVICE_URL",
    "http://fingerprint-service:8000",
)

ORCHESTRATION_SERVICE_URL = os.getenv(
    "ORCHESTRATION_SERVICE_URL",
    "http://orchestration-service:8080",
)

EVALUATION_USERNAME = os.getenv("EVALUATION_USERNAME", "evaluation-user")
EVALUATION_PASSWORD = os.getenv("EVALUATION_PASSWORD", "evaluation-password")

EVAL_DIR = Path(os.getenv("EVAL_DIR", "/app/eval"))
EVAL_QUERIES_DIR = Path(os.getenv("EVAL_QUERIES_DIR", str(EVAL_DIR / "queries")))

EVAL_QUERIES_PATH = Path(
    os.getenv("EVAL_QUERIES_PATH", str(EVAL_DIR / "eval_queries.jsonl"))
)

FINGERPRINT_RESULTS_PATH = Path(
    os.getenv("FINGERPRINT_RESULTS_PATH", str(EVAL_DIR / "fingerprint_results.jsonl"))
)

RECOGNITION_RESULTS_PATH = Path(
    os.getenv("RECOGNITION_RESULTS_PATH", str(EVAL_DIR / "recognition_results.jsonl"))
)

COMBINED_RESULTS_PATH = Path(
    os.getenv("COMBINED_RESULTS_PATH", str(EVAL_DIR / "combined_results.jsonl"))
)

METRICS_PATH = Path(
    os.getenv("METRICS_PATH", str(EVAL_DIR / "metrics.json"))
)

PREPARED_MANIFEST_PATH = Path(
    os.getenv(
        "PREPARED_MANIFEST_PATH",
        "/app/prepared-data/dataset_v1/manifest.jsonl",
    )
)

SR = int(os.getenv("EVAL_SR", "16000"))
DEFAULT_DURATIONS_SEC = [5.0, 10.0, 15.0]
DEFAULT_CORRUPTIONS = ["clean", "noisy", "strong_noisy", "phone_noisy"]

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minio")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minio123")
S3_BUCKET = os.getenv("S3_BUCKET", "tracks")

S3_PREFIX_CANDIDATES = [
    x.strip()
    for x in os.getenv("S3_PREFIX_CANDIDATES", ",fma/").split(",")
]

EVAL_AUDIO_SOURCE = os.getenv("EVAL_AUDIO_SOURCE", "s3").lower() # TODO switch to S3

DATASET_PROGRESS_EVERY = int(os.getenv("DATASET_PROGRESS_EVERY", "25"))
RUNNER_PROGRESS_EVERY = int(os.getenv("RUNNER_PROGRESS_EVERY", "25"))
