import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaylistEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):

        _, (h, _) = self.lstm(x)

        out = self.fc(h[-1])

        # 🔥 нормализация (ОЧЕНЬ ВАЖНО)
        out = F.normalize(out, dim=1)

        return out


class TrackEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):

        out = self.fc(x)

        # 🔥 нормализация
        out = F.normalize(out, dim=1)

        return out


class TwoTowerModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.playlist_encoder = PlaylistEncoder()
        self.track_encoder = TrackEncoder()

    def forward(self, playlist, track):

        p = self.playlist_encoder(playlist)
        t = self.track_encoder(track)

        # 🔥 cosine similarity
        return (p * t).sum(dim=1)
