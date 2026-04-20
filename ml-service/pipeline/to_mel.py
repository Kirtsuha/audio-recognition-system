import numpy as np
import librosa
import torch

from pipeline.config import SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, FMIN, FMAX


def to_mel(audio: np.ndarray, sr: int = SR) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.astype(np.float32)
    return torch.from_numpy(mel_db).unsqueeze(0)  # [1, n_mels, time]