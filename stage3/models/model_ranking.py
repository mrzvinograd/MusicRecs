import torch
import torch.nn as nn


class RankingModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, padding_idx=None, hidden_dim=128):
        super().__init__()

        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
        )

        self.playlist_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
        )

        self.track_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def encode_playlist(self, playlist):
        embedded = self.embedding(playlist)
        outputs, (hidden, _) = self.lstm(embedded)

        if self.padding_idx is None:
            mask = torch.ones_like(playlist, dtype=torch.bool)
        else:
            mask = playlist.ne(self.padding_idx)

        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = (outputs * mask).sum(dim=1) / denom
        last_hidden = hidden[-1]

        return self.playlist_proj(torch.cat([last_hidden, pooled], dim=1))

    def encode_track(self, track):
        return self.track_proj(self.embedding(track))

    def forward(self, playlist, track):
        playlist_vec = self.encode_playlist(playlist)
        track_vec = self.encode_track(track)

        if track_vec.dim() == 2:
            features = torch.cat(
                [
                    playlist_vec,
                    track_vec,
                    playlist_vec * track_vec,
                ],
                dim=1,
            )
            return self.fc(features).squeeze(-1)

        playlist_vec = playlist_vec.unsqueeze(1).expand(-1, track_vec.size(1), -1)
        features = torch.cat(
            [
                playlist_vec,
                track_vec,
                playlist_vec * track_vec,
            ],
            dim=2,
        )

        return self.fc(features).squeeze(-1)
