import torch
import torch.nn as nn


class TransformerModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=256):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):

        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        x = self.transformer(x)

        x = x[:, -1, :]  # берем последний токен

        logits = self.fc(x)  # (batch, vocab_size)

        return logits
