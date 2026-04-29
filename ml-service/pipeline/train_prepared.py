import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.model import AudioEncoder
from pipeline.augment import augment_audio
from pipeline.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    LR,
    EPOCHS,
    MARGIN,
    TRAIN_NEGATIVE_RETRIES,
    CACHE_PREPARED_IN_MEMORY,
    CACHE_PREPARED_MAX_TRACKS,
)
from pipeline.dataset import pad_or_trim, random_segment
from pipeline.manifest import iter_prepared_manifest
from pipeline.to_mel import to_mel_batch

logger = logging.getLogger("ml-pipeline.train")


class TripletLoss(nn.Module):
    def __init__(self, margin: float = MARGIN):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)


class PreparedTripletDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        limit: int | None = None,
        cache_in_memory: bool = CACHE_PREPARED_IN_MEMORY,
        cache_max_tracks: int = CACHE_PREPARED_MAX_TRACKS,
    ):
        items = [row for row in iter_prepared_manifest() if row["split"] == split]
        if limit is not None:
            items = items[:limit]

        if len(items) < 2:
            raise ValueError(f"Need at least 2 prepared items for split={split}, got {len(items)}")

        self.items = items
        self.cache_in_memory = cache_in_memory
        self.cache_max_tracks = cache_max_tracks
        self._cache: dict[str, np.ndarray] = {}

        logger.info(
            "PreparedTripletDataset initialized split=%s items=%s cache_in_memory=%s cache_max_tracks=%s",
            split,
            len(self.items),
            self.cache_in_memory,
            self.cache_max_tracks,
        )

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, path: str) -> np.ndarray:
        if not self.cache_in_memory:
            return np.load(path).astype("float32")

        cached = self._cache.get(path)
        if cached is not None:
            return cached

        audio = np.load(path).astype("float32")
        if self.cache_max_tracks <= 0 or len(self._cache) < self.cache_max_tracks:
            self._cache[path] = audio
        return audio

    def _load_segment(self, path: str) -> np.ndarray:
        audio = self._load_audio(path)
        return pad_or_trim(random_segment(audio))

    def __getitem__(self, idx: int):
        anchor_item = self.items[idx]
        audio = self._load_audio(anchor_item["prepared_path"])

        anchor = pad_or_trim(random_segment(audio))

        positive = pad_or_trim(random_segment(audio))
        positive = augment_audio(positive)

        negative_item = None
        for _ in range(TRAIN_NEGATIVE_RETRIES):
            candidate = random.choice(self.items)
            if candidate["track_id"] != anchor_item["track_id"]:
                negative_item = candidate
                break

        if negative_item is None:
            raise RuntimeError("Failed to sample negative item")

        negative = self._load_segment(negative_item["prepared_path"])
        if random.random() < 0.5:
            negative = augment_audio(negative)
        negative = pad_or_trim(negative)

        return anchor, positive, negative


def collate_to_mel(batch):
    anchors, positives, negatives = zip(*batch)

    a = torch.from_numpy(np.stack(anchors).astype("float32"))
    p = torch.from_numpy(np.stack(positives).astype("float32"))
    n = torch.from_numpy(np.stack(negatives).astype("float32"))

    return a, p, n


def train_prepared(
    output_model_path: str | Path,
    train_limit: int | None = None,
    epochs_override: int | None = None,
) -> None:
    started_total = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs_override if epochs_override is not None else EPOCHS

    dataset = PreparedTripletDataset(split="train", limit=train_limit)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_to_mel,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    logger.info(
        "Prepared training setup items=%s batch_size=%s epochs=%s num_workers=%s device=%s train_limit=%s",
        len(dataset),
        BATCH_SIZE,
        epochs,
        NUM_WORKERS,
        device,
        train_limit,
    )

    model = AudioEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = TripletLoss()

    for epoch in range(epochs):
        epoch_started = time.time()
        model.train()
        running_loss = 0.0

        logger.info("Epoch started epoch=%s/%s", epoch + 1, epochs)

        for batch_idx, (a, p, n) in enumerate(loader, start=1):
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            n = n.to(device, non_blocking=True)

            a_mel = to_mel_batch(a, device=device, normalize=True)
            p_mel = to_mel_batch(p, device=device, normalize=True)
            n_mel = to_mel_batch(n, device=device, normalize=True)

            emb_a = model(a_mel)
            emb_p = model(p_mel)
            emb_n = model(n_mel)

            loss = loss_fn(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0 or batch_idx == len(loader):
                avg = running_loss / batch_idx
                logger.info(
                    "Epoch progress epoch=%s/%s step=%s/%s avg_loss=%.4f",
                    epoch + 1,
                    epochs,
                    batch_idx,
                    len(loader),
                    avg,
                )

        epoch_loss = running_loss / max(len(loader), 1)
        logger.info(
            "Epoch finished epoch=%s/%s final_loss=%.4f elapsed_sec=%.2f",
            epoch + 1,
            epochs,
            epoch_loss,
            time.time() - epoch_started,
        )

    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model_path)

    logger.info(
        "Prepared training finished model_path=%s total_elapsed_sec=%.2f",
        output_model_path,
        time.time() - started_total,
    )