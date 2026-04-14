import torch
import torch.nn as nn

class AudioEncoder(nn.Module):

    def __init__(self, emb_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, dim=1)