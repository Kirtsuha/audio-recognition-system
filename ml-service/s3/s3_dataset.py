import os
import random

import librosa
from torch.utils.data import Dataset

from pipeline.augment import augment_audio
from pipeline.config import SR
from pipeline.dataset import random_segment
from s3.s3_loader import download_song


class S3TripletDataset(Dataset):
    def __init__(self, bucket: str, keys: list[str]):
        if not keys:
            raise ValueError("keys must not be empty")

        self.bucket = bucket
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def _load_random_segment(self, key: str):
        path = download_song(self.bucket, key)
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
            return random_segment(audio)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def __getitem__(self, idx: int):
        key_anchor = self.keys[idx]

        anchor = self._load_random_segment(key_anchor)
        positive = augment_audio(anchor)

        neg_idx = random.randrange(len(self.keys))
        while neg_idx == idx:
            neg_idx = random.randrange(len(self.keys))

        key_negative = self.keys[neg_idx]
        negative = self._load_random_segment(key_negative)
        negative = augment_audio(negative) if random.random() < 0.5 else negative

        return anchor, positive, negative