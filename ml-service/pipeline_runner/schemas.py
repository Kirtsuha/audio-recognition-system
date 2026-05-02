from pydantic import BaseModel, Field


class AsyncJobResponse(BaseModel):
    accepted: bool
    status: str
    bucket: str
    prefix: str


class ExperimentRunRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket")
    prefix: str = Field(default="", description="S3 prefix")

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
    index_all_prepared: bool = False

    fixed_eval_set_name: str = Field(default="default", min_length=1, max_length=100)
    eval_noise_mode: str = Field(default="both")  # clean | noisy | phone_noisy | both | all

    select_best_checkpoint: bool = False
    best_checkpoint_metric: str = "recall_at_1"
    best_checkpoint_eval_noise_mode: str = "noisy"

    aggregation_strategies: list[str] = Field(default_factory=lambda: ["current"])


class FullRunRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket")
    prefix: str = Field(default="", description="S3 prefix")

    prepare_limit: int = Field(default=0, ge=0)
    skip_prepare: bool = False

    train_limit: int = Field(default=8000, ge=0)
    val_limit: int = Field(default=1000, ge=0)
    test_limit: int = Field(default=1000, ge=0)

    index_all_prepared: bool = True
    index_windows_override: int = Field(default=96, ge=1, le=256)

    epochs_override: int = Field(default=10, ge=1, le=100)

    eval_query_limit: int = Field(default=500, ge=0)
    fixed_eval_set_name: str = Field(default="prod_val500_v1", min_length=1, max_length=100)
    eval_noise_mode: str = Field(default="noisy")  # clean | noisy | both
    aggregation_strategies: list[str] = Field(
        default_factory=lambda: ["max", "support", "hybrid_v2", "current"]
    )

    use_full_query_audio: bool = False

    select_best_checkpoint: bool = True
    best_checkpoint_metric: str = "recall_at_1"
    best_checkpoint_eval_noise_mode: str = "noisy"

    promote: bool = False


class EvaluateExistingModelRequest(BaseModel):
    model_path: str
    output_metrics_path: str | None = None

    train_limit: int = Field(default=8000, ge=0)
    val_limit: int = Field(default=1000, ge=0)
    test_limit: int = Field(default=1000, ge=0)

    eval_query_limit: int = Field(default=500, ge=0)
    index_windows_override: int = Field(default=96, ge=1, le=256)
    use_full_query_audio: bool = False

    fixed_eval_set_name: str = "prod_val500_v1"
    eval_noise_mode: str = "noisy"
    aggregation_strategies: list[str] = Field(
        default_factory=lambda: ["max", "support", "hybrid_v2", "current"]
    )

class IncrementalSyncRequest(BaseModel):
    bucket: str = Field(..., description="S3 bucket")
    prefix: str = Field(default="", description="S3 prefix")

    max_new_tracks: int = Field(default=0, ge=0)

    promote: bool = False
    allow_incremental_when_retrain_recommended: bool = True

    index_windows_override: int | None = Field(default=96, ge=1, le=256)


class PromoteRunRequest(BaseModel):
    run_dir: str
    require_metrics: bool = True


class CleanupRequest(BaseModel):
    dry_run: bool = True

    keep_last_full_runs: int = Field(default=2, ge=0)
    keep_last_experiment_runs: int = Field(default=5, ge=0)
    keep_last_incremental_runs: int = Field(default=2, ge=0)

    delete_failed_runs: bool = True
    delete_epoch_checkpoints: bool = True
    delete_eval_results: bool = False

    min_age_hours: int = Field(default=24, ge=0)


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
    aggregation_strategy: str | None = "max"


class BuildEmbeddingsRequest(BaseModel):
    model_path: str
    run_dir: str

    embeddings_filename: str = "embeddings.npy"
    song_ids_filename: str = "song_ids.npy"
    manifest_filename: str = "song_manifest.json"

    index_all_prepared: bool = True

    train_limit: int = Field(default=0, ge=0)
    val_limit: int = Field(default=0, ge=0)
    test_limit: int = Field(default=0, ge=0)

    index_windows_override: int | None = Field(default=96, ge=1, le=256)


class BuildFaissOnlyRequest(BaseModel):
    run_dir: str

    embeddings_filename: str = "embeddings.npy"
    song_ids_filename: str = "song_ids.npy"
    index_filename: str = "faiss.index"
    meta_filename: str = "faiss_meta.json"

    index_type: str | None = None  # flat | ivf_flat | None => from config
    nlist: int | None = Field(default=None, ge=1)
    nprobe: int | None = Field(default=None, ge=1)


class EvaluateArtifactsRequest(BaseModel):
    model_path: str
    index_path: str
    song_ids_path: str
    output_metrics_path: str

    faiss_meta_path: str | None = None
    output_results_path: str | None = None

    train_limit: int = Field(default=8000, ge=0)
    val_limit: int = Field(default=1000, ge=0)
    test_limit: int = Field(default=1000, ge=0)

    eval_query_limit: int = Field(default=500, ge=0)
    use_full_query_audio: bool = False

    fixed_eval_set_name: str = "prod_val500_v1"
    eval_noise_mode: str = "noisy"  # clean | noisy | phone_noisy | both | all

    aggregation_strategies: list[str] = Field(
        default_factory=lambda: ["max"]
    )