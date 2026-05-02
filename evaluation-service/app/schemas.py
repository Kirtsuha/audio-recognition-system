from pydantic import BaseModel, Field


class GenerateEvalDatasetRequest(BaseModel):
    per_track_clean_queries: int = Field(default=1, ge=0)
    per_track_noisy_queries: int = Field(default=2, ge=0)
    durations_sec: list[float] = Field(default_factory=lambda: [5.0, 10.0, 15.0])
    test_limit: int | None = Field(default=None, ge=1)
    negative_limit: int = Field(default=200, ge=0)


class EvalRunRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1)
    timeout_sec: float = Field(default=60.0, ge=1.0)


class RunAllRequest(BaseModel):
    generate_dataset: bool = True
    per_track_clean_queries: int = Field(default=1, ge=0)
    per_track_noisy_queries: int = Field(default=2, ge=0)
    durations_sec: list[float] = Field(default_factory=lambda: [5.0, 10.0, 15.0])
    test_limit: int | None = Field(default=None, ge=1)
    negative_limit: int = Field(default=200, ge=0)
    eval_limit: int | None = Field(default=None, ge=1)
    timeout_sec: float = Field(default=60.0, ge=1.0)