import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.model import AudioEncoder
from pipeline.config import MODEL_PATH, BATCH_SIZE, NUM_WORKERS, LR, EPOCHS, MARGIN
from pipeline.dataset import pad_or_trim
from pipeline.to_mel import to_mel
from s3.s3_dataset import S3TripletDataset
from s3.s3_list import list_all_songs

logger = logging.getLogger("ml-service.train")


class TripletLoss(nn.Module):
    def __init__(self, margin: float = MARGIN):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)


def collate_to_mel(batch):
    anchors, positives, negatives = zip(*batch)

    anchors = [pad_or_trim(x) for x in anchors]
    positives = [pad_or_trim(x) for x in positives]
    negatives = [pad_or_trim(x) for x in negatives]

    a = torch.stack([to_mel(x) for x in anchors])
    p = torch.stack([to_mel(x) for x in positives])
    n = torch.stack([to_mel(x) for x in negatives])

    return a, p, n


def train(bucket: str, prefix: str) -> None:
    started_total = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keys = list_all_songs(bucket=bucket, prefix=prefix)
    if len(keys) < 2:
        raise ValueError("Need at least two songs in S3 to train")

    logger.info(
        "Training setup bucket=%s prefix=%s songs=%s batch_size=%s epochs=%s num_workers=%s device=%s",
        bucket,
        prefix,
        len(keys),
        BATCH_SIZE,
        EPOCHS,
        NUM_WORKERS,
        device,
    )

    dataset = S3TripletDataset(bucket=bucket, keys=keys)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_to_mel,
        pin_memory=False,
        persistent_workers=False,
    )

    model = AudioEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = TripletLoss()

    logger.info("Training started total_batches_per_epoch=%s", len(loader))

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
        epoch_elapsed = time.time() - epoch_started

        logger.info(
            "Epoch finished epoch=%s/%s final_loss=%.4f elapsed_sec=%.2f",
            epoch + 1,
            EPOCHS,
            epoch_loss,
            epoch_elapsed,
        )

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    total_elapsed = time.time() - started_total
    logger.info("Training finished model_path=%s total_elapsed_sec=%.2f", MODEL_PATH, total_elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="fma/", help="S3 prefix")
    args = parser.parse_args()

    train(bucket=args.bucket, prefix=args.prefix)


if __name__ == "__main__":
    main()