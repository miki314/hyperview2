import torch
import torch.nn as nn
from neural_networks.Convolutional import CNN2D, CNN3D

class RNN(nn.Module):
    def __init__(self, neurons=256, hidden_size = 128, has_airborne_images = False):
        super().__init__()

        model_msi_2D = CNN2D(channels=13, neurons=neurons)
        model_msi_2D.load_state_dict(torch.load("models/nn/CNN2D_MSI.pt"))

        # model_msi_3D = CNN3D(neurons=neurons)
        # model_msi_3D.load_state_dict(torch.load("models/nn/CNN3D_MSI.pt"))

        model_hsi = CNN2D(channels=231, neurons=neurons)
        model_hsi.load_state_dict(torch.load("models/nn/CNN2D_HSI.pt"))

        if has_airborne_images:
            model_aerial = CNN2D(channels=421, neurons=neurons)
            model_aerial.load_state_dict(torch.load("models/nn/CNN2D_airborne.pt"))

        self.encoders = nn.ModuleList([
            model_msi_2D,
            # model_msi_3D,
            model_hsi,
            model_aerial
        ] if has_airborne_images else [
            model_msi_2D,
            # model_msi_3D,
            model_hsi
        ])

        #RNN, LSTM, 
        self.rnn = nn.RNN(
            input_size=neurons,
            hidden_size=2048,
            batch_first=True
        )
        
        self.lstm= nn.LSTM(
            input_size=neurons,
            hidden_size=2048,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 6),
            nn.LeakyReLU()
        )

    def forward(self, inputs):
        feats = []

        for x, enc in zip(inputs, self.encoders):
            f = enc.extract_features(x) 
            feats.append(f)

        seq = torch.stack(feats, dim=1)

        out, _ = self.rnn(seq)
        last = out[:, -1, :]

        return self.fc(last)

class BandRNN(nn.Module):
    def __init__(
        self,
        in_bands = 13,
        compressed_channels=32,
        rnn_size = 64,
        hidden_size=128,
        rnn_hidden_size=128,
        output_dim=6,
        rnn_type = "RNN"
    ):
        super().__init__()

        self.spectral_reduce = nn.Conv2d(
            in_channels=in_bands,
            out_channels=compressed_channels,
            kernel_size=1
        )
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (batch, 64, 1, 1)
        )

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=rnn_size,
                hidden_size=rnn_hidden_size,
                batch_first=True
            )
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=rnn_size,
                hidden_size=rnn_hidden_size,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                input_size=rnn_size,
                hidden_size=rnn_hidden_size,
                batch_first=True
            )

        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 6),
        )

    def forward(self, x):
        x = self.spectral_reduce(x)  # (B, C, H, W)
    
        B, C, H, W = x.shape
    
        x = x.view(B * C, 1, H, W)
    
        feat = self.spatial_encoder(x)   # (B*C, 64, 1, 1)
        feat = feat.view(B, C, 64)       # (B, C, 64)
        
        out, _ = self.rnn(feat)
        out = out[:, -1, :]
    
        return self.fc(out)
    
class FusionNN(nn.Module):
    def __init__(self, neurons=256, hidden_size=128, has_airborne_images=False):
        super().__init__()

        model_msi_2D = CNN2D(channels=13, neurons=neurons)
        model_msi_2D.load_state_dict(torch.load("models/nn/CNN2D_MSI.pt"))
        
        # model_msi_3D = CNN3D(neurons=neurons)
        # model_msi_3D.load_state_dict(torch.load("models/nn/CNN3D_MSI.pt"))

        model_hsi = CNN2D(channels=231, neurons=neurons) # 3 for PCA
        model_hsi.load_state_dict(torch.load("models/nn/CNN2D_HSI.pt"))

        encoders = [model_msi_2D, model_hsi]

        if has_airborne_images:
            model_aerial = CNN2D(channels=421, neurons=neurons)
            model_aerial.load_state_dict(torch.load("models/nn/CNN2D_airborne.pt"))
            encoders.append(model_aerial)

        self.encoders = nn.ModuleList(encoders)

        self.num_encoders = len(encoders)

        self.fc = nn.Sequential(
            nn.Linear(neurons * self.num_encoders, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, 6)
        )

    def forward(self, inputs):
        feats = []

        for x, enc in zip(inputs, self.encoders):
            f = enc.extract_features(x)  # (B, neurons)
            feats.append(f)

        fused = torch.cat(feats, dim=1)  # (B, neurons * num_encoders)

        return self.fc(fused)
