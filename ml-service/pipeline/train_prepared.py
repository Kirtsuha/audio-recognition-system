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
from pipeline.config import BATCH_SIZE, NUM_WORKERS, LR, EPOCHS, MARGIN, TRAIN_NEGATIVE_RETRIES
from pipeline.dataset import iter_prepared_manifest, pad_or_trim, random_segment
from pipeline.to_mel import to_mel

logger = logging.getLogger("ml-pipeline.train")


class TripletLoss(nn.Module):
    def __init__(self, margin: float = MARGIN):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)


class PreparedTripletDataset(Dataset):
    def __init__(self, split: str = "train"):
        self.items = [row for row in iter_prepared_manifest() if row["split"] == split]
        if len(self.items) < 2:
            raise ValueError(f"Need at least 2 prepared items for split={split}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_segment(self, path: str) -> np.ndarray:
        audio = np.load(path).astype("float32")
        return pad_or_trim(random_segment(audio))

    def __getitem__(self, idx: int):
        anchor_item = self.items[idx]
        anchor = self._load_segment(anchor_item["prepared_path"])

        positive = augment_audio(anchor)
        positive = pad_or_trim(positive)

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

    a = torch.stack([to_mel(x) for x in anchors])
    p = torch.stack([to_mel(x) for x in positives])
    n = torch.stack([to_mel(x) for x in negatives])

    return a, p, n


def train_prepared(output_model_path: str | Path) -> None:
    started_total = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PreparedTripletDataset(split="train")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_to_mel,
        pin_memory=False,
        persistent_workers=False,
    )

    logger.info(
        "Prepared training setup items=%s batch_size=%s epochs=%s num_workers=%s device=%s",
        len(dataset),
        BATCH_SIZE,
        EPOCHS,
        NUM_WORKERS,
        device,
    )

    model = AudioEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = TripletLoss()

    for epoch in range(EPOCHS):
        epoch_started = time.time()
        model.train()
        running_loss = 0.0

        logger.info("Epoch started epoch=%s/%s", epoch + 1, EPOCHS)

        for batch_idx, (a, p, n) in enumerate(loader, start=1):
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)

            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

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
                    EPOCHS,
                    batch_idx,
                    len(loader),
                    avg,
                )

        epoch_loss = running_loss / max(len(loader), 1)
        logger.info(
            "Epoch finished epoch=%s/%s final_loss=%.4f elapsed_sec=%.2f",
            epoch + 1,
            EPOCHS,
            epoch_loss,
            time.time() - epoch_started,
        )

    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model_path)

    logger.info("Prepared training finished model_path=%s total_elapsed_sec=%.2f", output_model_path, time.time() - started_total)