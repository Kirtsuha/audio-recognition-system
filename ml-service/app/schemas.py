from typing import Any

from pydantic import BaseModel, Field


class RecognitionResponse(BaseModel):
    matched: bool
    song_id: int | None = None
    score: float | None = None
    confidence: float | None = None
    margin: float | None = None
    support: int | None = None
    support_ratio: float | None = None
    reason: str | None = None
    top_candidates: list[dict[str, Any]] = Field(default_factory=list)


class ReloadResponse(BaseModel):
    reloaded: bool
    runtime_ready: bool


class PipelineRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket")
    prefix: str = Field(default="fma/", description="S3 prefix")


class AsyncJobResponse(BaseModel):
    accepted: bool
    status: str
    bucket: str
    prefix: str

class ExperimentRunRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket")
    prefix: str = Field(default="fma/", description="S3 prefix")

    train_limit: int = Field(default=0, ge=0, description="0 means no limit")
    val_limit: int = Field(default=0, ge=0, description="0 means no limit")
    test_limit: int = Field(default=0, ge=0, description="0 means no limit")

    epochs_override: int | None = Field(default=None, ge=1, le=100)
    eval_query_limit: int = Field(default=0, ge=0, description="0 means no limit")
    index_windows_override: int | None = Field(default=None, ge=1, le=256)

    experiment_name: str = Field(default="experiment", min_length=1, max_length=100)
    use_full_query_audio: bool = False

class EvalGenerateRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket")
    prefix: str = Field(default="fma/", description="S3 prefix")
    per_track_clean_queries: int = Field(default=2, ge=1, le=20)
    per_track_noisy_queries: int = Field(default=4, ge=1, le=20)
    per_track_negative_queries: int = Field(default=0, ge=0, le=20)
    durations_sec: list[float] = Field(default_factory=lambda: [3.0, 5.0, 8.0])


class EvalRunRequest(BaseModel):
    warmup_queries: int = Field(default=3, ge=0, le=100)
    limit: int | None = Field(default=None, ge=1)


class ThresholdTuneRequest(BaseModel):
    confidence_values: list[float] = Field(default_factory=lambda: [0.50, 0.55, 0.58, 0.60, 0.65, 0.70])
    margin_values: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.20, 0.50, 1.0])
    support_values: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])