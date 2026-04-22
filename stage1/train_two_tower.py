import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import TWO_TOWER_MODEL_PT
from stage1.dataset_playlist import PlaylistDataset
from stage1.evaluate_two_tower import recall_at_k
from stage1.models.two_tower import TwoTowerModel


# 🔥 CONFIG
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4
MAX_STEPS = 10000  # ограничение на эпоху (чтобы не ждать вечность)


# 🔥 DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 🔥 DATASET
dataset = PlaylistDataset()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE
)


# 🔥 MODEL
model = TwoTowerModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()


# 🚀 TRAIN LOOP
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for step, (playlist, target) in enumerate(loader):

        if step > MAX_STEPS:
            break

        playlist = playlist.to(device)
        target = target.to(device)

        # --- positive ---
        pos_score = model(playlist, target)

        # --- negative sampling ---
        # берём embedding playlist
        with torch.no_grad():
            p_vec = model.playlist_encoder(playlist)

        # случайные кандидаты
        rand_idx = torch.randint(0, dataset.embeddings.shape[0], (1000,))
        candidates = torch.tensor(dataset.embeddings[rand_idx], dtype=torch.float32).to(device)

        # считаем similarity
        scores = torch.matmul(p_vec, candidates.T)

        # берём САМЫЕ ПОХОЖИЕ (hard negatives)
        hard_idx = torch.topk(scores, k=target.size(0), dim=1).indices

        neg = candidates[hard_idx[0]]
        neg_score = model(playlist, neg)

        # --- loss ---
        loss = (
            loss_fn(pos_score, torch.ones_like(pos_score)) +
            loss_fn(neg_score, torch.zeros_like(neg_score))
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 500 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item()}")

    print(f"\n🔥 Epoch {epoch} Total Loss {total_loss}\n")


# 💾 SAVE MODEL
torch.save(model.state_dict(), TWO_TOWER_MODEL_PT)


# 📊 EVALUATION
print("Evaluating Recall@10...")

recall = recall_at_k(
    model,
    dataset,
    dataset.embeddings,
    K=10,
    num_samples=1000
)

print(f"\n🎯 Recall@10: {recall}\n")
