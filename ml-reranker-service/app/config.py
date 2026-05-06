from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = "ml-reranker-service"


    ref_embeddings_path: str = "/app/models/active/reranker_ref_embeddings.npy"
    ref_meta_path: str = "/app/models/active/reranker_ref_meta.jsonl"
    ref_config_path: str = "/app/models/active/reranker_ref_config.json"

    use_precomputed_reference_embeddings: bool = True
    ref_neighbor_windows: int = 1


    ref_fallback_to_audio: bool = True


    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str = "minio"
    s3_secret_key: str = "minio123"
    s3_region: str = "us-east-1"

    query_audio_bucket: str = "recognition-queries"
    reference_audio_bucket: str = "full-tracks"
    reference_audio_fallback_buckets: str = "tracks,full-tracks"


    model_path: str = "/app/models/model.pt"
    emb_dim: int = 128


    sr: int = 16000
    segment_seconds: float = 15.0


    n_mels: int = 96
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    fmin: float = 20.0
    fmax: float | None = None


    max_candidates: int = 20
    fingerprint_offset_hop_seconds: float = 1.0

    offset_jitter_seconds: str = "-0.75,0.0,0.75"

    fp_weight: float = 0.95
    ml_weight: float = 0.05

    no_match_threshold: float = 0.45


    class Config:
        env_prefix = "ML_RERANKER_"
        env_file = ".env"


settings = Settings()
