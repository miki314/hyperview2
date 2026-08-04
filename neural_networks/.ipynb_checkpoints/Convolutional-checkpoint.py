import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self, kernel_size=3, neurons=128, padding=1, channels=13, out_channels=6):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, neurons, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),

            nn.Conv2d(neurons, neurons, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),

            nn.Conv2d(neurons, neurons, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),

            nn.Conv2d(neurons, neurons, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),

            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.shared_fc = nn.Sequential(
            nn.Linear(neurons* 4, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU()
        )

        self.heads = nn.ModuleList([
            nn.Linear(neurons, 1) for _ in range(out_channels)
        ])

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        features = self.shared_fc(x)

        outputs = [head(features) for head in self.heads]
        return torch.cat(outputs, dim=1)

    def extract_features(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.shared_fc(x)
        
class CNN3D(nn.Module):
    def __init__(self, out_channels=6, neurons = 128, kernel_size = 3, channels=13):
        super().__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv3d(1, neurons, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(),

            nn.Conv3d(neurons, neurons, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(),

            nn.Conv3d(neurons, neurons, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(),
            
            nn.MaxPool3d(kernel_size=(1,2,2)),

            nn.Conv3d(neurons, neurons, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(),

            nn.Conv3d(neurons, neurons, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(),

            nn.Conv3d(neurons, neurons, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(),

            nn.MaxPool3d(kernel_size=(1,2,2)),
            nn.AdaptiveAvgPool3d((1,1,1))
        )

        self.shared_fc = nn.Sequential(
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU()
        )

        self.heads = nn.ModuleList([
            nn.Linear(neurons, 1) for _ in range(out_channels)
        ])

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, D, H, W)

        x = self.shared_conv(x)          # (B, 128, 1, 1, 1)
        x = x.view(x.size(0), -1)        # (B, 128)

        features = self.shared_fc(x)     # (B, 128)

        outputs = [head(features) for head in self.heads]
        return torch.cat(outputs, dim=1) # (B, 6)
        # x = self.conv3d(x)
        # x = x.view(x.size(0), -1)

        # return self.fc(x)
    def extract_features(self, x):
        x = self.shared_conv(x)
        x = x.view(x.size(0), -1)
        return self.shared_fc(x)