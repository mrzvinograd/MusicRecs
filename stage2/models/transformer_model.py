import torch
import torch.nn as nn


class TransformerModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=256, max_seq_len=20, padding_idx=None):

        super().__init__()

        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden = self.embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)

        padding_mask = None
        if self.padding_idx is not None:
            padding_mask = input_ids.eq(self.padding_idx)

        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)

        if padding_mask is None:
            pooled = hidden[:, -1, :]
        else:
            valid_lengths = (~padding_mask).sum(dim=1).clamp(min=1)
            last_positions = (valid_lengths - 1).unsqueeze(1).unsqueeze(2)
            pooled = hidden.gather(1, last_positions.expand(-1, 1, hidden.size(-1))).squeeze(1)

        return self.fc(pooled)
