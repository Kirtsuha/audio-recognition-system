import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np

from pipeline.config import FAISS_INDEX_TYPE, FAISS_NLIST, FAISS_NPROBE

logger = logging.getLogger("ml-pipeline.index-build")


def _safe_nlist(n_vectors: int, requested_nlist: int) -> int:
    if n_vectors < 1024:
        return max(1, min(requested_nlist, n_vectors))

    return max(16, min(requested_nlist, n_vectors // 8))


def _write_meta(
    meta_out_path: str | Path,
    *,
    index_type: str,
    vectors: int,
    dim: int,
    nlist: int | None = None,
    nprobe: int | None = None,
) -> None:
    meta_out_path = Path(meta_out_path)
    meta_out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "index_type": index_type,
        "vectors": int(vectors),
        "dim": int(dim),
        "nlist": int(nlist) if nlist is not None else None,
        "nprobe": int(nprobe) if nprobe is not None else None,
        "metric": "inner_product",
        "normalized_embeddings": True,
    }

    with meta_out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_faiss_index(
    embeddings_path: str | Path,
    song_ids_path: str | Path,
    index_out_path: str | Path,
    meta_out_path: str | Path | None = None,
    index_type: str | None = None,
    nlist: int | None = None,
    nprobe: int | None = None,
) -> dict:
    started = time.time()

    embeddings = np.load(embeddings_path).astype("float32")
    song_ids = np.load(song_ids_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {embeddings.shape}")

    if len(embeddings) != len(song_ids):
        raise ValueError("embeddings and song_ids have different lengths")

    if len(embeddings) == 0:
        raise ValueError("Cannot build FAISS index from empty embeddings")

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index_type = (index_type or FAISS_INDEX_TYPE).lower()
    nprobe = nprobe or FAISS_NPROBE

    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        effective_nlist = None

    elif index_type == "ivf_flat":
        requested_nlist = nlist or FAISS_NLIST
        effective_nlist = _safe_nlist(len(embeddings), requested_nlist)

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantizer,
            dim,
            effective_nlist,
            faiss.METRIC_INNER_PRODUCT,
        )

        logger.info(
            "Training IVF index vectors=%s dim=%s nlist=%s",
            len(embeddings),
            dim,
            effective_nlist,
        )

        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe

    else:
        raise ValueError(f"Unsupported FAISS_INDEX_TYPE={index_type}")

    index_out_path = Path(index_out_path)
    index_out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_out_path))

    if meta_out_path is not None:
        _write_meta(
            meta_out_path,
            index_type=index_type,
            vectors=index.ntotal,
            dim=dim,
            nlist=effective_nlist,
            nprobe=nprobe if index_type == "ivf_flat" else None,
        )

    summary = {
        "index_type": index_type,
        "vectors": int(index.ntotal),
        "dim": int(dim),
        "nlist": int(effective_nlist) if effective_nlist is not None else None,
        "nprobe": int(nprobe) if index_type == "ivf_flat" else None,
        "index_path": str(index_out_path),
        "meta_path": str(meta_out_path) if meta_out_path is not None else None,
        "elapsed_sec": time.time() - started,
    }

    logger.info("FAISS index build finished summary=%s", summary)
    return summary


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

    if not index.is_trained:
        raise RuntimeError("Cannot append to untrained FAISS index")

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