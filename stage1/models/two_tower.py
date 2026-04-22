import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaylistEncoder(nn.Module):

    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return F.normalize(out, dim=1)


class TrackEncoder(nn.Module):

    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        out = self.fc(x)
        return F.normalize(out, dim=1)


class TwoTowerModel(nn.Module):

    def __init__(self, embed_dim=512, hidden_dim=256):
        super().__init__()

        self.playlist_encoder = PlaylistEncoder(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
        )
        self.track_encoder = TrackEncoder(
            input_dim=embed_dim,
            output_dim=embed_dim,
        )

    def encode_playlist(self, playlist):
        return self.playlist_encoder(playlist)

    def encode_tracks(self, tracks):
        return self.track_encoder(tracks)

    def forward(self, playlist, track):
        p = self.encode_playlist(playlist)
        t = self.encode_tracks(track)
        return (p * t).sum(dim=1)
