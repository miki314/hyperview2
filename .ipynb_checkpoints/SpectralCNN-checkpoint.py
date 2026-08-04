import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

def load_image_data(directory):
    files = sorted(directory.glob('*.npz'))
    msi_data = []
    for file in enumerate(files):
        file_name = file[1]
        with np.load(file_name) as npz:
            arr = np.ma.MaskedArray(**npz)
            msi_data.append(arr)

    return msi_data

class SpectralCNN(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x)

def masked_mse(pred, target, mask):

    diff = (pred - target) ** 2
    diff = diff * mask

    loss = diff.sum() / mask.sum().clamp(min=1)

    return loss
