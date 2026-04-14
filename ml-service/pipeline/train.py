import torch
import librosa
import torch.nn as nn
from torch.utils.data import DataLoader

from app.model import AudioEncoder
from pipeline.config import MODEL_PATH
from s3.s3_dataset import S3TripletDataset
from pipeline.to_mel import to_mel

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos = (anchor - positive).pow(2).sum(1)
        neg = (anchor - negative).pow(2).sum(1)
        return torch.relu(pos - neg + self.margin).mean()


def batch_to_mel(batch):

    anchors, positives, negatives = [], [], []

    for a, p, n in batch:
        anchors.append(to_mel(a))
        positives.append(to_mel(p))
        negatives.append(to_mel(n))

    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
    )



device = "cuda" if torch.cuda.is_available() else "cpu"

model = AudioEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = TripletLoss()

dataset = S3TripletDataset(bucket, keys)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for epoch in range(10):

    # loss = train_step(
    #     song_a=songs[0],
    #     song_b=songs[1]
    # )
    for batch in loader:
        a, p, n = batch_to_mel(batch)

        a, p, n = a.to(device), p.to(device), n.to(device)

        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)

        loss = loss_fn(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




    print(f"epoch {epoch}, loss = {loss}")

torch.save(model.state_dict(), MODEL_PATH)




