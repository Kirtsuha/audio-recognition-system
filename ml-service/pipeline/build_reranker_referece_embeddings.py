from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from app.model import AudioEncoder
from pipeline.config import (
    EMB_DIM,
    RERANKER_REF_BATCH_SIZE,
    RERANKER_REF_HOP_SECONDS,
    RERANKER_REF_WINDOW_SECONDS,
    SR,
)
from pipeline.manifest import load_all_prepared_manifest, load_prepared_manifest
from pipeline.to_mel import to_mel_batch

logger = logging.getLogger("ml-pipeline.reranker-ref-build")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / norms).astype("float32")


def _pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    audio = np.asarray(audio, dtype="float32")

    if len(audio) == target_len:
        return audio

    if len(audio) > target_len:
        return audio[:target_len].astype("float32")

    out = np.zeros(target_len, dtype="float32")
    out[: len(audio)] = audio
    return out


def extract_reference_windows(
    audio: np.ndarray,
    *,
    window_seconds: float,
    hop_seconds: float,
) -> list[tuple[np.ndarray, float]]:
    """
    Returns:
        [(window_audio, start_sec), ...]
    """
    window_len = int(round(window_seconds * SR))
    hop_len = int(round(hop_seconds * SR))

    if window_len <= 0:
        raise ValueError(f"Invalid window_seconds={window_seconds}")

    if hop_len <= 0:
        raise ValueError(f"Invalid hop_seconds={hop_seconds}")

    audio = np.asarray(audio, dtype="float32")

    if len(audio) <= window_len:
        return [(_pad_or_trim(audio, window_len), 0.0)]

    max_start = len(audio) - window_len
    starts = list(range(0, max_start + 1, hop_len))

    if not starts or starts[-1] != max_start:
        starts.append(max_start)

    return [
        (
            audio[start:start + window_len].astype("float32"),
            float(start / SR),
        )
        for start in starts
    ]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_reranker_reference_embeddings(
    *,
    model_path: str | Path,
    embeddings_out: str | Path,
    meta_out: str | Path,
    config_out: str | Path,
    train_limit: int = 0,
    val_limit: int = 0,
    test_limit: int = 0,
    only_track_ids: set[int] | None = None,
    index_all_prepared: bool = True,
    window_seconds: float = RERANKER_REF_WINDOW_SECONDS,
    hop_seconds: float = RERANKER_REF_HOP_SECONDS,
    batch_size: int = RERANKER_REF_BATCH_SIZE,
) -> dict:
    started = time.time()

    model_path = Path(model_path)
    embeddings_out = Path(embeddings_out)
    meta_out = Path(meta_out)
    config_out = Path(config_out)

    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")

    if index_all_prepared:
        items = load_all_prepared_manifest(only_track_ids=only_track_ids)
    else:
        items = load_prepared_manifest(
            train_limit=train_limit,
            val_limit=val_limit,
            test_limit=test_limit,
            only_track_ids=only_track_ids,
        )

    if not items:
        raise ValueError("Prepared manifest is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioEncoder().to(device)
    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    all_meta: list[dict] = []

    pending_windows: list[np.ndarray] = []
    pending_meta: list[dict] = []

    def flush() -> None:
        nonlocal pending_windows, pending_meta

        if not pending_windows:
            return

        audio_batch = np.stack(pending_windows).astype("float32")

        with torch.inference_mode():
            mel_batch = to_mel_batch(
                audio_batch,
                device=device,
                normalize=True,
            )
            embeddings = model(mel_batch).detach().cpu().numpy().astype("float32")

        embeddings = _l2_normalize(embeddings)

        all_embeddings.append(embeddings)
        all_meta.extend(pending_meta)

        pending_windows = []
        pending_meta = []

    total_windows = 0

    for item_idx, row in enumerate(items, start=1):
        track_id = int(row["track_id"])
        s3_key = str(row["s3_key"])
        prepared_path = str(row["prepared_path"])

        audio = np.load(prepared_path).astype("float32")
        windows = extract_reference_windows(
            audio,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )

        if item_idx <= 5:
            logger.info(
                "Reranker ref item track_id=%s audio_sec=%.2f windows=%s window_sec=%.2f hop_sec=%.2f",
                track_id,
                len(audio) / SR,
                len(windows),
                window_seconds,
                hop_seconds,
            )

        for window_idx, (window, start_sec) in enumerate(windows):
            embedding_index = total_windows

            pending_windows.append(window)
            pending_meta.append(
                {
                    "embedding_index": int(embedding_index),
                    "track_id": int(track_id),
                    "s3_key": s3_key,
                    "prepared_path": prepared_path,
                    "split": row.get("split"),
                    "window_idx": int(window_idx),
                    "start_sec": float(start_sec),
                    "window_seconds": float(window_seconds),
                    "hop_seconds": float(hop_seconds),
                    "duration_sec": float(row.get("duration_sec", len(audio) / SR)),
                }
            )

            total_windows += 1

            if len(pending_windows) >= batch_size:
                flush()

        if item_idx % 250 == 0 or item_idx == len(items):
            flush()
            logger.info(
                "Reranker ref build progress processed=%s/%s vectors=%s",
                item_idx,
                len(items),
                total_windows,
            )

    flush()

    if not all_embeddings:
        raise RuntimeError("No embeddings were built")

    embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")

    if embeddings.ndim != 2:
        raise RuntimeError(f"Invalid embeddings shape: {embeddings.shape}")

    if embeddings.shape[1] != EMB_DIM:
        raise RuntimeError(f"Invalid embedding dim: got={embeddings.shape[1]} expected={EMB_DIM}")

    if len(all_meta) != len(embeddings):
        raise RuntimeError(f"Meta/embeddings mismatch: meta={len(all_meta)} embeddings={len(embeddings)}")

    embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.parent.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_out, embeddings)
    _write_jsonl(meta_out, all_meta)

    config = {
        "model_path": str(model_path),
        "embeddings_path": str(embeddings_out),
        "meta_path": str(meta_out),
        "emb_dim": int(embeddings.shape[1]),
        "vectors": int(len(embeddings)),
        "tracks": int(len(items)),
        "sr": int(SR),
        "window_seconds": float(window_seconds),
        "hop_seconds": float(hop_seconds),
        "batch_size": int(batch_size),
        "index_all_prepared": bool(index_all_prepared),
        "train_limit": int(train_limit),
        "val_limit": int(val_limit),
        "test_limit": int(test_limit),
    }

    with config_out.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    summary = {
        **config,
        "elapsed_sec": float(time.time() - started),
    }

    logger.info("Reranker reference embeddings built summary=%s", summary)
    return summary