import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import torch

from app.model import AudioEncoder
from pipeline.config import (
    FAISS_INDEX_PATH,
    MIN_CONFIDENCE,
    MIN_MARGIN,
    MIN_SUPPORTED_WINDOWS,
    MODEL_PATH,
    SONG_IDS_PATH,
)
from pipeline.to_mel import to_mel

logger = logging.getLogger("ml-service.runtime")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
runtime_lock = threading.RLock()


@dataclass(frozen=True)
class RuntimeArtifacts:
    model: AudioEncoder
    index: faiss.Index
    song_ids: np.ndarray
    device: torch.device


_model: AudioEncoder | None = None
_index: faiss.Index | None = None
_song_ids: np.ndarray | None = None


def load_runtime_artifacts() -> None:
    global _model, _index, _song_ids

    with runtime_lock:
        missing = []

        if not os.path.exists(MODEL_PATH):
            missing.append(str(MODEL_PATH))
        if not os.path.exists(FAISS_INDEX_PATH):
            missing.append(str(FAISS_INDEX_PATH))
        if not os.path.exists(SONG_IDS_PATH):
            missing.append(str(SONG_IDS_PATH))

        if missing:
            logger.warning("Runtime artifacts missing, degraded mode enabled: %s", missing)
            _model = None
            _index = None
            _song_ids = None
            return

        loaded_model = AudioEncoder().to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        loaded_model.load_state_dict(state)
        loaded_model.eval()

        loaded_index = faiss.read_index(str(FAISS_INDEX_PATH))
        loaded_song_ids = np.load(SONG_IDS_PATH)

        if loaded_index.ntotal != len(loaded_song_ids):
            raise RuntimeError(
                f"Inconsistent runtime artifacts: index.ntotal={loaded_index.ntotal}, "
                f"song_ids={len(loaded_song_ids)}"
            )

        _model = loaded_model
        _index = loaded_index
        _song_ids = loaded_song_ids

        logger.info(
            "Runtime loaded successfully device=%s vectors=%s dim=%s",
            device,
            loaded_index.ntotal,
            loaded_index.d,
        )


def runtime_ready() -> bool:
    return _model is not None and _index is not None and _song_ids is not None


def ensure_runtime_ready() -> None:
    if not runtime_ready():
        raise RuntimeError("ML runtime is not ready. Build and promote artifacts first, then reload runtime.")


def get_runtime() -> RuntimeArtifacts:
    ensure_runtime_ready()

    assert _model is not None
    assert _index is not None
    assert _song_ids is not None

    return RuntimeArtifacts(
        model=_model,
        index=_index,
        song_ids=_song_ids,
        device=device,
    )


def should_accept(result: dict[str, Any]) -> bool:
    return (
        result["confidence"] >= MIN_CONFIDENCE
        and result["margin"] >= MIN_MARGIN
        and result["support"] >= MIN_SUPPORTED_WINDOWS
    )


def embed_windows(windows: list[np.ndarray]) -> np.ndarray:
    runtime = get_runtime()

    batch = torch.stack([to_mel(w) for w in windows]).to(runtime.device)

    with torch.inference_mode():
        embs = runtime.model(batch).detach().cpu().numpy().astype("float32")

    faiss.normalize_L2(embs)
    return embs


def runtime_status() -> dict[str, Any]:
    return {
        "runtime_ready": runtime_ready(),
        "model_exists": os.path.exists(MODEL_PATH),
        "index_exists": os.path.exists(FAISS_INDEX_PATH),
        "song_ids_exists": os.path.exists(SONG_IDS_PATH),
        "device": str(device),
        "index_total": _index.ntotal if _index is not None else None,
        "song_ids_total": len(_song_ids) if _song_ids is not None else None,
        "index_dim": _index.d if _index is not None else None,
        "model_dim": _model.head[-1].out_features if _model is not None else None,
    }