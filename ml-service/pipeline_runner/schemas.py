from pydantic import BaseModel, Field

class EvalGenerateRequest(BaseModel):
    per_track_clean_queries: int = Field(default=2, ge=1, le=20)
    per_track_noisy_queries: int = Field(default=4, ge=1, le=20)
    durations_sec: list[float] = Field(default_factory=lambda: [5.0, 8.0, 15.0])


class EvalRunRequest(BaseModel):
    warmup_queries: int = Field(default=3, ge=0, le=100)
    limit: int | None = Field(default=None, ge=1)


class ThresholdTuneRequest(BaseModel):
    confidence_values: list[float] = Field(default_factory=lambda: [0.50, 0.55, 0.58, 0.60, 0.65, 0.70])
    margin_values: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.20, 0.50, 1.0])
    support_values: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])

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

    train_limit: int = Field(default=0, ge=0)
    val_limit: int = Field(default=0, ge=0)
    test_limit: int = Field(default=0, ge=0)
    prepare_limit: int = Field(default=0, ge=0)
    skip_prepare: bool = False

    epochs_override: int | None = Field(default=None, ge=1, le=100)
    eval_query_limit: int = Field(default=0, ge=0)
    index_windows_override: int | None = Field(default=None, ge=1, le=256)
    use_full_query_audio: bool = False

    experiment_name: str = Field(default="experiment", min_length=1, max_length=100)
    build_artifacts: bool = False