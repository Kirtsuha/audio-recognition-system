import faiss
import numpy as np

embeddings = np.load("embeddings.npy")
song_ids = np.load("song_ids.npy")

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)

index.add(embeddings)

faiss.write_index(index, "faiss.index")