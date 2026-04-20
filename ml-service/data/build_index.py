import faiss
import numpy as np

from pipeline.config import EMBEDDINGS_PATH, SONG_IDS_PATH, FAISS_INDEX_PATH


def main() -> None:
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    song_ids = np.load(SONG_IDS_PATH)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {embeddings.shape}")

    if len(embeddings) != len(song_ids):
        raise ValueError("embeddings.npy and song_ids.npy have different lengths")

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Saved index to {FAISS_INDEX_PATH}")
    print(f"Vectors: {index.ntotal}, dim: {dim}")


if __name__ == "__main__":
    main()