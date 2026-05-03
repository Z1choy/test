import torch.nn as nn


class PointOffsetHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        bottleneck_dim = max(hidden_dim // 2, 8)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)
