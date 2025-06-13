import torch
import torch.nn as nn

class DraftNet(nn.Module):
    def __init__(self, hero_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hero_count, 512), nn.SiLU(),
            nn.Linear(512, 256),            nn.SiLU(),
            nn.Linear(256, hero_count)
        )

    def forward(self, x, mask):
        logits = self.net(x)
        return logits.masked_fill(mask.bool(), -1e9)
