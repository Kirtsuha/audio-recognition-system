from fastapi import FastAPI, UploadFile
import faiss
import numpy as np
import torch

from app.model import AudioEncoder
from app.inference import aggregate_results
from pipeline.config import MODEL_PATH
from pipeline.dataset import load_audio, random_segment
from pipeline.to_mel import to_mel

app = FastAPI()

model = AudioEncoder()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

index = faiss.read_index("data/faiss.index")
song_ids = np.load("data/song_ids.npy")


@app.post("/recognize")
async def recognize(file: UploadFile):

    audio_bytes = await file.read()

    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    audio = load_audio("temp.wav")

    all_indices = []

    for _ in range(5):
        seg = random_segment(audio)
        mel = to_mel(seg).unsqueeze(0)

        emb = model(mel).detach().numpy().astype("float32")

        D, I = index.search(emb, k=5)
        all_indices.append(I[0])

    result = aggregate_results(all_indices, song_ids)

    return {"song_id": int(result)}