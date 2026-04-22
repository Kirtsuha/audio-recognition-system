import logging
import time
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger("ml-pipeline.index-build")


def build_faiss_index(
    embeddings_path: str | Path,
    song_ids_path: str | Path,
    index_out_path: str | Path,
) -> None:
    started = time.time()

    embeddings = np.load(embeddings_path).astype("float32")
    song_ids = np.load(song_ids_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {embeddings.shape}")

    if len(embeddings) != len(song_ids):
        raise ValueError("embeddings and song_ids have different lengths")

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_out_path = Path(index_out_path)
    index_out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_out_path))

    logger.info(
        "FAISS index build finished vectors=%s dim=%s elapsed_sec=%.2f path=%s",
        index.ntotal,
        dim,
        time.time() - started,
        index_out_path,
    )


def append_to_faiss_index(
    base_index_path: str | Path,
    new_embeddings_path: str | Path,
    index_out_path: str | Path,
) -> None:
    started = time.time()

    index = faiss.read_index(str(base_index_path))
    new_embeddings = np.load(new_embeddings_path).astype("float32")

    if new_embeddings.ndim != 2:
        raise ValueError(f"Expected new embeddings shape [N, D], got {new_embeddings.shape}")

    faiss.normalize_L2(new_embeddings)
    index.add(new_embeddings)

    index_out_path = Path(index_out_path)
    index_out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_out_path))

    logger.info(
        "FAISS index append finished added=%s total=%s elapsed_sec=%.2f path=%s",
        len(new_embeddings),
        index.ntotal,
        time.time() - started,
        index_out_path,
    )