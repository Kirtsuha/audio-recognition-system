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