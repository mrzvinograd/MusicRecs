import sys
from pathlib import Path

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FILTERED_TRACK_ID_MAP_PKL, PLAYLIST_TRACKS_PARQUET, RANKING_MODEL_PT
from stage3.dataset_ranking import RankingDataset, ranking_collate
from stage3.evaluate_ranking import evaluate_ranking
from stage3.models.model_ranking import RankingModel


BATCH_SIZE = 64
EPOCHS = 15
MAX_STEPS = 15000
NUM_NEG = 8
DB_PATH = str(PLAYLIST_TRACKS_PARQUET)
EVAL_MOD = 20
EVAL_REMAINDER = 0
USE_HARD_NEGATIVES = True
TRAIN_HARD_NEGATIVE_RATIO = 0.4
EVAL_HARD_NEGATIVE_RATIO = 0.7
HARD_NEGATIVE_POOL_SIZE = 128


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


with open(FILTERED_TRACK_ID_MAP_PKL, "rb") as f:
    track_map = pickle.load(f)

vocab_size = len(track_map)
pad_idx = vocab_size


dataset = RankingDataset(
    db_path=DB_PATH,
    track_map=track_map,
    num_neg=NUM_NEG,
    eval_mod=EVAL_MOD,
    eval_remainder=EVAL_REMAINDER,
    use_hard_negatives=USE_HARD_NEGATIVES,
    hard_negative_ratio=TRAIN_HARD_NEGATIVE_RATIO,
    hard_pool_size=HARD_NEGATIVE_POOL_SIZE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    collate_fn=lambda batch: ranking_collate(batch, pad_idx),
)


model = RankingModel(vocab_size + 1, padding_idx=pad_idx).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    print("Streaming ranking data...")

    for step, (playlist, track, label) in enumerate(loader):
        playlist = playlist.to(device)
        track = track.to(device)
        label = label.to(device)

        score = model(playlist, track)
        loss = loss_fn(score, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 500 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

        if step >= MAX_STEPS:
            break

    print(f"\nEpoch {epoch} Total Loss {total_loss:.4f}")

    metrics = evaluate_ranking(
        model=model,
        db_path=DB_PATH,
        track_map=track_map,
        pad_idx=pad_idx,
        k_values=(10,),
        num_candidates=200,
        num_samples=1000,
        batch_size=32,
        eval_mod=EVAL_MOD,
        eval_remainder=EVAL_REMAINDER,
        use_hard_negatives=USE_HARD_NEGATIVES,
        hard_negative_ratio=EVAL_HARD_NEGATIVE_RATIO,
        hard_pool_size=HARD_NEGATIVE_POOL_SIZE,
    )

    print(
        "Validation "
        f"Recall@10={metrics['recall@10']:.4f} "
        f"NDCG@10={metrics['ndcg@10']:.4f} "
        f"MRR@10={metrics['mrr@10']:.4f}"
    )


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size + 1,
        "pad_idx": pad_idx,
        "model_kwargs": {
            "vocab_size": vocab_size + 1,
            "padding_idx": pad_idx,
        },
    },
    RANKING_MODEL_PT,
)

print(f"Saved {RANKING_MODEL_PT}")
