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