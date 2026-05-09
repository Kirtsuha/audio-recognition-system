import sys
import types


class FakeMelSpectrogram:
    def __init__(self, *args, n_mels=96, **kwargs):
        self.n_mels = n_mels

    def to(self, _device):
        return self

    def __call__(self, audio_batch):
        import torch

        batch_size = audio_batch.shape[0]
        frames = max(1, audio_batch.shape[-1] // 256)
        return torch.ones((batch_size, self.n_mels, frames), dtype=torch.float32, device=audio_batch.device)


fake_torchaudio = types.ModuleType("torchaudio")
fake_torchaudio.transforms = types.SimpleNamespace(MelSpectrogram=FakeMelSpectrogram)
sys.modules.setdefault("torchaudio", fake_torchaudio)
