import torch.nn as nn


class PointOffsetHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, output_activation="tanh"):
        super().__init__()
        bottleneck_dim = max(hidden_dim // 2, 8)
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, 1),
        ]
        if output_activation.lower() == "tanh":
            layers.append(nn.Tanh())
        elif output_activation.lower() != "linear":
            raise ValueError(f"Unsupported point offset activation: {output_activation}")
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
