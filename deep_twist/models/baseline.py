import torch
import torch.nn as nn


class Random(nn.Module):
    def __init__(self):
        super(Random, self).__init__()

    def forward(self):
        return torch.FloatTensor((320, 240, 0, 50, 50))
