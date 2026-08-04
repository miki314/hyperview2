import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_features=12, h1=64, h2=64, h3=64, output_features=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, h3),
            nn.ReLU(),
            nn.Linear(h3, output_features)
        )

    def forward(self, x):
        return self.model(x)