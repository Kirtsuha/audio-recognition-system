from pathlib import Path
from typing import List

import numpy as np
import torch

from app.model import AudioEncoder
from pipeline.config import MODEL_PATH, INDEX_WINDOWS_PER_SONG
from pipeline.dataset import load_audio, extract_uniform_index_windows
from pipeline.to_mel import to_mel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> AudioEncoder:
    model = AudioEncoder().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def embed_windows(model: AudioEncoder, windows: List[np.ndarray]) -> np.ndarray:
    batch = torch.stack([to_mel(window) for window in windows]).to(device)

    with torch.inference_mode():
        embs = model(batch).cpu().numpy().astype("float32")

    return embs


def embed_song(path: str | Path, n_windows: int = INDEX_WINDOWS_PER_SONG) -> np.ndarray:
    model = load_model()
    audio = load_audio(str(path))
    windows = extract_uniform_index_windows(audio, n_windows=n_windows)
    return embed_windows(model, windows)