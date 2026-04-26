import os
import random

import librosa
from torch.utils.data import Dataset

from pipeline.augment import augment_audio
from pipeline.config import SR
from pipeline.dataset import random_segment, pad_or_trim
from s3.s3_loader import download_song


class S3TripletDataset(Dataset):
    def __init__(self, bucket: str, keys: list[str], max_decode_retries: int = 10):
        if len(keys) < 2:
            raise ValueError("Need at least two keys for triplet dataset")

        self.bucket = bucket
        self.keys = keys
        self.max_decode_retries = max_decode_retries

    def __len__(self) -> int:
        return len(self.keys)

    def _load_audio_segment_once(self, key: str):
        path = download_song(self.bucket, key)

        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Downloaded path does not exist: {path}")

            if os.path.getsize(path) == 0:
                raise ValueError(f"Downloaded file is empty: key={key}")

            audio, _ = librosa.load(path, sr=SR, mono=True)

            if audio is None or len(audio) == 0:
                raise ValueError(f"Decoded empty audio for key={key}")

            segment = random_segment(audio)
            return pad_or_trim(segment)

        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _load_audio_segment(self, preferred_key: str | None = None):
        """
        Try preferred key first, then fall back to random keys.
        This prevents one bad/corrupted file from crashing the whole training.
        """
        candidate_keys = []
        if preferred_key is not None:
            candidate_keys.append(preferred_key)

        for _ in range(self.max_decode_retries - len(candidate_keys)):
            candidate_keys.append(random.choice(self.keys))

        last_exc = None

        for key in candidate_keys:
            try:
                return self._load_audio_segment_once(key)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

        raise RuntimeError("Failed to decode any audio segment after multiple retries") from last_exc

    def __getitem__(self, idx: int):
        anchor_key = self.keys[idx]

        anchor = self._load_audio_segment(preferred_key=anchor_key)

        positive = augment_audio(anchor)
        positive = pad_or_trim(positive)

        neg_idx = random.randrange(len(self.keys))
        while neg_idx == idx:
            neg_idx = random.randrange(len(self.keys))

        negative_key = self.keys[neg_idx]
        negative = self._load_audio_segment(preferred_key=negative_key)

        if random.random() < 0.5:
            negative = augment_audio(negative)

        negative = pad_or_trim(negative)

        return anchor, positive, negative