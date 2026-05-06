from __future__ import annotations

import numpy as np
import torch
import torchaudio

from app.config import settings

_MEL_TRANSFORMS: dict[str, torchaudio.transforms.MelSpectrogram] = {}


def _get_mel_transform(device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    key = str(device)

    if key not in _MEL_TRANSFORMS:
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=settings.sr,
            n_fft=settings.n_fft,
            win_length=settings.win_length,
            hop_length=settings.hop_length,
            f_min=settings.fmin,
            f_max=settings.fmax if settings.fmax is not None else settings.sr // 2,
            n_mels=settings.n_mels,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        ).to(device)

        _MEL_TRANSFORMS[key] = transform

    return _MEL_TRANSFORMS[key]


def power_to_db_like_librosa(mel: torch.Tensor, amin: float = 1e-10) -> torch.Tensor:
    mel = torch.clamp(mel, min=amin)
    log_spec = 10.0 * torch.log10(mel)
    ref = torch.amax(log_spec, dim=(-2, -1), keepdim=True)
    mel_db = log_spec - ref
    mel_db = torch.clamp(mel_db, min=-80.0, max=0.0)
    return mel_db


def normalize_mel_db(mel_db: torch.Tensor) -> torch.Tensor:
    return (mel_db + 80.0) / 80.0


def to_mel_batch(
    audio_batch: torch.Tensor | np.ndarray,
    device: torch.device | str | None = None,
    normalize: bool = True,
) -> torch.Tensor:


    if device is None:
        if isinstance(audio_batch, torch.Tensor):
            device = audio_batch.device
        else:
            device = torch.device("cpu")

    device = torch.device(device)

    if isinstance(audio_batch, np.ndarray):
        audio_batch = torch.from_numpy(audio_batch.astype("float32"))

    audio_batch = audio_batch.to(device=device, dtype=torch.float32)

    if audio_batch.ndim == 1:
        audio_batch = audio_batch.unsqueeze(0)

    if audio_batch.ndim != 2:
        raise ValueError(f"Expected audio batch shape [B, samples], got {tuple(audio_batch.shape)}")

    mel_transform = _get_mel_transform(device)
    mel = mel_transform(audio_batch)
    mel_db = power_to_db_like_librosa(mel)

    if normalize:
        mel_db = normalize_mel_db(mel_db)

    return mel_db.unsqueeze(1).contiguous()