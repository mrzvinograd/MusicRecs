import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import TWO_TOWER_MODEL_PT
from stage1.dataset_playlist import PlaylistDataset
from stage1.evaluate_two_tower import evaluate_two_tower
from stage1.models.two_tower import TwoTowerModel


DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-4
DEFAULT_MAX_STEPS = 30000
DEFAULT_NEGATIVE_CANDIDATE_POOL = 1024
DEFAULT_EVAL_SAMPLES = 1000
DEFAULT_TEMPERATURE = 0.07
DEFAULT_MIN_LR = 1e-6
DEFAULT_LR_PATIENCE = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage1 two-tower retrieval model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--negative-candidate-pool", type=int, default=DEFAULT_NEGATIVE_CANDIDATE_POOL)
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument("--lr-patience", type=int, default=DEFAULT_LR_PATIENCE)
    return parser.parse_args()


def sample_negative_indices(total_tracks, candidate_pool_size, exclude_indices, device):
    sample_size = min(candidate_pool_size, max(0, total_tracks - len(exclude_indices)))

    if sample_size <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    negatives = set()

    while len(negatives) < sample_size:
        needed = sample_size - len(negatives)
        candidates = torch.randint(0, total_tracks, (needed * 2,), device=device).tolist()

        for candidate in candidates:
            if candidate in exclude_indices:
                continue

            negatives.add(candidate)

            if len(negatives) >= sample_size:
                break

    return torch.tensor(list(negatives), dtype=torch.long, device=device)


args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(
    "Training config:",
    f"batch_size={args.batch_size}",
    f"epochs={args.epochs}",
    f"max_steps={args.max_steps}",
    f"negative_candidate_pool={args.negative_candidate_pool}",
    f"eval_samples={args.eval_samples}",
    f"temperature={args.temperature}",
    f"min_lr={args.min_lr}",
    f"lr_patience={args.lr_patience}",
)

train_dataset = PlaylistDataset(return_target_idx=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

eval_dataset = PlaylistDataset(return_target_idx=True)


model = TwoTowerModel(embed_dim=train_dataset.feature_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=args.lr_patience,
    min_lr=args.min_lr,
)
best_recall = float("-inf")


for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0

    for step, (playlist, target) in enumerate(train_loader):
        if step >= args.max_steps:
            break

        playlist = playlist.to(device)
        target = target.to(device)

        playlist_vec = model.encode_playlist(playlist)
        target_vec = model.encode_tracks(target)

        rand_idx = sample_negative_indices(
            train_dataset.embeddings.shape[0],
            args.negative_candidate_pool,
            exclude_indices=set(),
            device=device,
        )
        if rand_idx.numel() > 0:
            negative_candidates = torch.tensor(
                train_dataset.get_track_features(rand_idx.tolist()),
                dtype=torch.float32,
                device=device,
            )
            negative_vec = model.encode_tracks(negative_candidates)
            negative_logits = torch.matmul(playlist_vec, negative_vec.T) / args.temperature
        else:
            negative_logits = torch.empty((playlist.size(0), 0), device=device)

        # Retrieval training: each playlist should rank its own target above
        # both other targets in the batch and a shared sampled negative pool.
        positive_logits = torch.matmul(playlist_vec, target_vec.T) / args.temperature
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        labels = torch.arange(playlist.size(0), device=device)

        loss_playlist_to_track = F.cross_entropy(logits, labels)
        loss_track_to_playlist = F.cross_entropy(positive_logits.T, labels)
        loss = 0.5 * (loss_playlist_to_track + loss_track_to_playlist)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 500 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    print(f"\nEpoch {epoch} Total Loss {total_loss:.4f}")

    metrics = evaluate_two_tower(
        model=model,
        dataset=eval_dataset,
        k_values=(10, 50),
        num_samples=args.eval_samples,
        candidate_pool_size=args.negative_candidate_pool,
    )

    print(
        "Validation "
        f"Recall@10={metrics['recall@10']:.4f} "
        f"NDCG@10={metrics['ndcg@10']:.4f} "
        f"MRR@10={metrics['mrr@10']:.4f}"
    )
    scheduler.step(metrics["recall@10"])

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Learning rate: {current_lr:.8f}")

    if metrics["recall@10"] > best_recall:
        best_recall = metrics["recall@10"]
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_kwargs": {
                    "embed_dim": train_dataset.feature_dim,
                    "hidden_dim": 256,
                },
                "metrics": metrics,
                "epoch": epoch,
            },
            TWO_TOWER_MODEL_PT,
        )
        print(f"Saved best checkpoint to {TWO_TOWER_MODEL_PT}")
