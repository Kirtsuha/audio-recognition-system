from __future__ import annotations

from pydantic import BaseModel, Field

class MinioAudioRef(BaseModel):
    bucket: str
    key: str

class FingerprintTopOffset(BaseModel):
    offset: int | float
    aligned_matches: int


class FingerprintCandidate(BaseModel):
    track_id: int | str
    best_offset: int | float
    aligned_matches: int = 0
    total_matches: int = 0
    offset_count: int = 0
    coverage: float = 0.0
    unique_coverage: float = 0.0
    top_offsets: list[FingerprintTopOffset] = Field(default_factory=list)
    score_gap: float = 0.0
    confidence: float = 0.0
    title: str | None = None
    artist: str | None = None
    s3_key: str


    best_offset_sec: float | None = None


    model_config = {
        "extra": "allow",
    }


class FingerprintRecognitionPayload(BaseModel):
    source: str = "fingerprint"
    top_k: int = 0
    query_hashes: int = 0
    unique_query_hashes: int = 0
    best_aligned_matches: int = 0
    second_aligned_matches: int = 0
    reason: str | None = None
    candidates: list[FingerprintCandidate] = Field(default_factory=list)

    model_config = {
        "extra": "allow",
    }


class RerankOptions(BaseModel):
    max_candidates: int | None = None
    segment_seconds: float | None = None
    fingerprint_offset_hop_seconds: float | None = None
    offset_jitter_seconds: list[float] | None = None
    fp_weight: float | None = None
    ml_weight: float | None = None
    no_match_threshold: float | None = None


class RerankRequest(BaseModel):
    request_id: str
    query_audio: MinioAudioRef
    reference_bucket: str | None = None
    fingerprint: FingerprintRecognitionPayload
    options: RerankOptions = Field(default_factory=RerankOptions)


class RerankedCandidate(BaseModel):
    track_id: int | str
    title: str | None = None
    artist: str | None = None
    s3_key: str

    best_offset: int | float
    best_offset_sec: float

    fingerprint_confidence: float
    fingerprint_score: float
    ml_similarity: float
    ml_probability: float
    final_confidence: float

    aligned_matches: int
    total_matches: int
    offset_count: int
    coverage: float
    unique_coverage: float
    score_gap: float


class RerankResponse(BaseModel):
    request_id: str
    source: str = "ml_reranker"
    matched: bool
    reason: str | None = None
    best: RerankedCandidate | None = None
    candidates: list[RerankedCandidate] = Field(default_factory=list)
    timing_ms: dict[str, int] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    device: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    model_path: str
    query_audio_bucket: str
    reference_audio_bucket: str
    sr: int
    segment_seconds: float
    max_candidates: int
    reference_store_loaded: bool = False
    reference_store_vectors: int = 0
    reference_store_tracks: int = 0


