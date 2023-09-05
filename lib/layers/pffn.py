import torch.nn as nn
import torch.nn.functional as F


class PointWiseFFN(nn.Module):
    def __init__(self, d_hidden: int, d_ff: int):
        super().__init__()

        self.dense1 = nn.Linear(d_hidden, d_ff)
        self.dense2 = nn.Linear(d_ff, d_hidden)

    def forward(self, h):
        h = self.dense1(h)
        h = F.relu(h)
        h = self.dense2(h)

        return h
