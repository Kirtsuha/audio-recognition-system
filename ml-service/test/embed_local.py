import numpy as np

from pipeline.embed import embed_song
from pipeline.config import EMBEDDINGS_PATH, SONG_IDS_PATH

songs = [
    ("../data/songs/1.wav", 1),
    ("../data/songs/2.wav", 2),
    ("../data/songs/3.wav", 3),
]

all_embeddings = []
all_ids = []

for path, song_id in songs:
    embs = embed_song(path)
    for emb in embs:
        all_embeddings.append(emb)
        all_ids.append(song_id)

embeddings = np.asarray(all_embeddings, dtype="float32")
song_ids = np.asarray(all_ids, dtype=np.int64)

np.save(EMBEDDINGS_PATH, embeddings)
np.save(SONG_IDS_PATH, song_ids)

print(f"Saved {len(embeddings)} embeddings")