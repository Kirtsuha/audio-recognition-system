import numpy as np

from pipeline.embed import embed_song

songs = [
    ("../data/songs/1.wav", 1),
    ("../data/songs/2.wav", 2),
    ("../data/songs/3.wav", 3),
]

all_embeddings = []
all_ids = []

for path, song_id in songs:
    embs = embed_song(path)
    for e in embs:
        all_embeddings.append(e)
        all_ids.append(song_id)

np.save("../data/embeddings.npy", np.array(all_embeddings))
np.save("../data/song_ids.npy", np.array(all_ids))