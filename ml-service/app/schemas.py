from typing import Any

from pydantic import BaseModel, Field


class TrainFromS3Request(BaseModel):
    bucket: str = Field(..., description="S3 bucket name")
    prefix: str = Field(default="fma/", description="S3 prefix")


class AsyncJobResponse(BaseModel):
    accepted: bool
    status: str
    bucket: str
    prefix: str


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