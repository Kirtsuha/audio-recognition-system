from __future__ import annotations

import bisect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("ml-reranker.reference-store")


@dataclass(frozen=True)
class ReferenceWindow:
    embedding_index: int
    track_id: int
    s3_key: str
    start_sec: float
    window_idx: int
    window_seconds: float
    hop_seconds: float


class ReferenceEmbeddingStore:
    def __init__(
        self,
        embeddings_path: str | Path,
        meta_path: str | Path,
        config_path: str | Path | None = None,
    ):
        self.embeddings_path = Path(embeddings_path)
        self.meta_path = Path(meta_path)
        self.config_path = Path(config_path) if config_path else None

        self.embeddings: np.ndarray | None = None
        self.track_windows: dict[int, list[ReferenceWindow]] = {}
        self.track_starts: dict[int, list[float]] = {}
        self.config: dict[str, Any] | None = None

    def loaded(self) -> bool:
        return self.embeddings is not None and bool(self.track_windows)

    def load(self) -> None:
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Reference embeddings not found: {self.embeddings_path}")

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Reference meta not found: {self.meta_path}")

        embeddings = np.load(self.embeddings_path, mmap_mode="r")

        if embeddings.ndim != 2:
            raise RuntimeError(f"Invalid embeddings shape: {embeddings.shape}")

        track_windows: dict[int, list[ReferenceWindow]] = {}

        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)

                window = ReferenceWindow(
                    embedding_index=int(row["embedding_index"]),
                    track_id=int(row["track_id"]),
                    s3_key=str(row.get("s3_key") or ""),
                    start_sec=float(row["start_sec"]),
                    window_idx=int(row.get("window_idx", 0)),
                    window_seconds=float(row.get("window_seconds", 15.0)),
                    hop_seconds=float(row.get("hop_seconds", 2.0)),
                )

                track_windows.setdefault(window.track_id, []).append(window)

        for track_id in list(track_windows.keys()):
            track_windows[track_id].sort(key=lambda item: item.start_sec)

        track_starts = {
            track_id: [window.start_sec for window in windows]
            for track_id, windows in track_windows.items()
        }

        config = None
        if self.config_path and self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)

        self.embeddings = embeddings
        self.track_windows = track_windows
        self.track_starts = track_starts
        self.config = config

        logger.info(
            "Reference embedding store loaded embeddings=%s tracks=%s meta=%s config=%s",
            tuple(embeddings.shape),
            len(track_windows),
            self.meta_path,
            self.config_path,
        )

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded(),
            "embeddings_path": str(self.embeddings_path),
            "meta_path": str(self.meta_path),
            "config_path": str(self.config_path) if self.config_path else None,
            "vectors": int(self.embeddings.shape[0]) if self.embeddings is not None else 0,
            "dim": int(self.embeddings.shape[1]) if self.embeddings is not None else None,
            "tracks": len(self.track_windows),
            "config": self.config,
        }

    def get_nearby_embeddings(
        self,
        *,
        track_id: int,
        offset_sec: float,
        neighbor_windows: int = 1,
    ) -> tuple[np.ndarray, list[ReferenceWindow]] | None:
        if self.embeddings is None:
            return None

        windows = self.track_windows.get(int(track_id))
        starts = self.track_starts.get(int(track_id))

        if not windows or not starts:
            return None

        offset_sec = max(float(offset_sec), 0.0)

        insert_pos = bisect.bisect_left(starts, offset_sec)

        candidate_positions = set()

        for pos in [insert_pos - 1, insert_pos, insert_pos + 1]:
            if 0 <= pos < len(windows):
                candidate_positions.add(pos)

        nearest_pos = None
        nearest_distance = float("inf")

        for pos in candidate_positions:
            dist = abs(starts[pos] - offset_sec)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_pos = pos

        if nearest_pos is None:
            return None

        left = max(0, nearest_pos - int(neighbor_windows))
        right = min(len(windows), nearest_pos + int(neighbor_windows) + 1)

        selected_windows = windows[left:right]
        indices = [window.embedding_index for window in selected_windows]

        if not indices:
            return None

        embs = np.asarray(self.embeddings[indices], dtype="float32")
        return embs, selected_windows