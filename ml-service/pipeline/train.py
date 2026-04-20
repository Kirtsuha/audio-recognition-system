import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.model import AudioEncoder
from pipeline.config import MODEL_PATH, BATCH_SIZE, NUM_WORKERS, LR, EPOCHS, MARGIN
from pipeline.to_mel import to_mel
from s3.s3_dataset import S3TripletDataset
from s3.s3_list import list_all_songs


class TripletLoss(nn.Module):
    def __init__(self, margin: float = MARGIN):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)


def collate_to_mel(batch):
    anchors, positives, negatives = zip(*batch)

    a = torch.stack([to_mel(x) for x in anchors])
    p = torch.stack([to_mel(x) for x in positives])
    n = torch.stack([to_mel(x) for x in negatives])

    return a, p, n


def train(bucket: str, prefix: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keys = list_all_songs(bucket=bucket, prefix=prefix)
    if len(keys) < 2:
        raise ValueError("Need at least two songs in S3 to train")

    dataset = S3TripletDataset(bucket=bucket, keys=keys)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_to_mel,
        pin_memory=torch.cuda.is_available(),
    )

    model = AudioEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = TripletLoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (a, p, n) in enumerate(loader, start=1):
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            n = n.to(device, non_blocking=True)

            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = loss_fn(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                avg = running_loss / batch_idx
                print(f"epoch={epoch + 1}/{EPOCHS} step={batch_idx} loss={avg:.4f}")

        epoch_loss = running_loss / max(len(loader), 1)
        print(f"epoch={epoch + 1}/{EPOCHS} final_loss={epoch_loss:.4f}")

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="fma/", help="S3 prefix")
    args = parser.parse_args()

    train(bucket=args.bucket, prefix=args.prefix)


if __name__ == "__main__":
    main()