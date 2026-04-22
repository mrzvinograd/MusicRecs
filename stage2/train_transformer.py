import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import duckdb
import pickle

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FILTERED_TRACK_ID_MAP_PKL, PLAYLIST_TRACKS_PARQUET
from stage2.dataset_sequence import SequenceDataset
from stage2.models.transformer_model import TransformerModel


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# DUCKDB (ограничение RAM)
# =========================
print("Connecting to DuckDB...")

con = duckdb.connect()
con.execute("PRAGMA memory_limit='16GB'")


# =========================
# LOAD TRACK MAP
# =========================
print("Loading track map...")

with open(FILTERED_TRACK_ID_MAP_PKL, "rb") as f:
    track_map = pickle.load(f)

vocab_size = len(track_map)
print("Vocab size:", vocab_size)


# =========================
# DATASET
# =========================
dataset = SequenceDataset(
    seq_len=20,
    db_path=str(PLAYLIST_TRACKS_PARQUET),
    track_map=track_map
)

loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=0
)


# =========================
# MODEL
# =========================
model = TransformerModel(vocab_size=vocab_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


# =========================
# TRAIN
# =========================
for epoch in range(3):

    total_loss = 0

    print("Streaming playlists...")

    for step, (seq, target) in enumerate(loader):

        seq = seq.to(device)
        target = target.to(device)

        logits = model(seq)

        loss = loss_fn(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 500 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item()}")

        if step > 10000:
            break

    print(f"\n🔥 Epoch {epoch} Total Loss {total_loss}")


# =========================
# EVALUATION (Recall@10)
# =========================
print("\nEvaluating Recall@10...")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for step, (seq, target) in enumerate(loader):

        seq = seq.to(device)
        target = target.to(device)

        logits = model(seq)

        topk = torch.topk(logits, k=10, dim=1).indices

        match = (topk == target.unsqueeze(1)).any(dim=1)

        correct += match.sum().item()
        total += target.size(0)

        if step > 2000:
            break

recall = correct / total
print(f"\n🎯 Recall@10: {recall}")
