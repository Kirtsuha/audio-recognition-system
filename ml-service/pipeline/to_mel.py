import numpy as np
import torch
import torchaudio

from pipeline.config import SR, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, FMIN, FMAX

_MEL_TRANSFORMS: dict[str, torchaudio.transforms.MelSpectrogram] = {}


def _get_mel_transform(device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    key = str(device)

    if key not in _MEL_TRANSFORMS:
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            f_min=FMIN,
            f_max=FMAX,
            n_mels=N_MELS,
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
    """
    mel: [B, n_mels, frames]
    Аналог librosa.power_to_db(mel, ref=np.max), но batched и на torch.
    """
    mel = torch.clamp(mel, min=amin)

    log_spec = 10.0 * torch.log10(mel)

    ref = torch.amax(log_spec, dim=(-2, -1), keepdim=True)
    mel_db = log_spec - ref

    # примерно как типичный диапазон librosa power_to_db
    mel_db = torch.clamp(mel_db, min=-80.0, max=0.0)

    return mel_db


def normalize_mel_db(mel_db: torch.Tensor) -> torch.Tensor:
    """
    [-80, 0] -> [0, 1]
    """
    return (mel_db + 80.0) / 80.0


def to_mel_batch(
    audio_batch: torch.Tensor | np.ndarray,
    device: torch.device | str | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Быстрый batched mel.

    Input:
        audio_batch: [B, samples] или [samples]

    Output:
        [B, 1, n_mels, frames]
    """
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
    mel = mel_transform(audio_batch)  # [B, n_mels, frames]

    mel_db = power_to_db_like_librosa(mel)

    if normalize:
        mel_db = normalize_mel_db(mel_db)

    return mel_db.unsqueeze(1).contiguous()  # [B, 1, n_mels, frames]


# def to_mel(audio: np.ndarray, sr: int = SR) -> torch.Tensor:
#     """
#     Backward-compatible API для старого кода.
#     Возвращает [1, n_mels, frames].
#     """
#     if sr != SR:
#         raise ValueError(f"Expected sr={SR}, got sr={sr}")
#
#     with torch.no_grad():
#         batch = to_mel_batch(audio, device="cpu")
#         return batch[0]