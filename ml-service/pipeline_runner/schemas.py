from pydantic import BaseModel, Field

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

    fixed_eval_set_name: str = Field(default="default", min_length=1, max_length=100)
    eval_noise_mode: str = Field(default="both")  # clean | noisy | both
    select_best_checkpoint: bool = False
    best_checkpoint_metric: str = "recall_at_1"
    best_checkpoint_eval_noise_mode: str = "noisy"
    aggregation_strategies: list[str] = Field(
        default_factory=lambda: ["current"]
    )

class ThresholdTuneExperimentRequest(BaseModel):
    run_dir: str
    results_filename: str = "eval_results.jsonl"

    confidence_values: list[float] = Field(
        default_factory=lambda: [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    )
    margin_values: list[float] = Field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]
    )
    support_values: list[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5]
    )

    case: str | None = "noisy"
    aggregation_strategy: str | None = "current"


class FullRunRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket")
    prefix: str = Field(default="", description="S3 prefix")

    train_limit: int = Field(default=0, ge=0)
    val_limit: int = Field(default=0, ge=0)
    test_limit: int = Field(default=0, ge=0)
    prepare_limit: int = Field(default=0, ge=0)
    skip_prepare: bool = False

    epochs_override: int | None = Field(default=10, ge=1, le=100)
    eval_query_limit: int = Field(default=500, ge=0)
    index_windows_override: int | None = Field(default=96, ge=1, le=256)
    use_full_query_audio: bool = False

    fixed_eval_set_name: str = Field(default="prod_val500_v1", min_length=1, max_length=100)
    eval_noise_mode: str = Field(default="both")
    aggregation_strategies: list[str] = Field(default_factory=lambda: ["max"])

    promote: bool = True