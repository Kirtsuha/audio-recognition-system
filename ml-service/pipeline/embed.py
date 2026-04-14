import torch

from app.model import AudioEncoder
from pipeline.config import MODEL_PATH, N_SEGMENTS
from pipeline.dataset import load_audio, random_segment
from pipeline.to_mel import to_mel

model = AudioEncoder()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def embed_song(path, n_segments=N_SEGMENTS):

    audio = load_audio(path)

    vectors = []

    for _ in range(n_segments):
        seg = random_segment(audio)
        mel = to_mel(seg).unsqueeze(0)
        emb = model(mel).detach().numpy()[0]
        vectors.append(emb)

    return vectors