import numpy as np
import torch
import torchaudio

from pipeline.config import (
    SR,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    FMIN,
    FMAX,
    TORCHAUDIO_MEL_DEVICE,
)

_device = torch.device(TORCHAUDIO_MEL_DEVICE)

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    f_min=FMIN,
    f_max=FMAX,
    n_mels=N_MELS,
    power=2.0,
).to(_device)

_db_transform = torchaudio.transforms.AmplitudeToDB(stype="power").to(_device)


def to_mel(audio: np.ndarray | torch.Tensor, sr: int = SR) -> torch.Tensor:
    if sr != SR:
        raise ValueError(f"Unexpected sample rate: {sr}, expected {SR}")

    if isinstance(audio, np.ndarray):
        wav = torch.from_numpy(audio.astype(np.float32))
    elif isinstance(audio, torch.Tensor):
        wav = audio.float()
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    wav = wav.to(_device)

    if wav.ndim != 1:
        wav = wav.reshape(-1)

    mel = _mel_transform(wav)
    mel_db = _db_transform(mel)

    # always return on CPU for compatibility with current pipeline
    return mel_db.unsqueeze(0).cpu()  # [1, n_mels, time]