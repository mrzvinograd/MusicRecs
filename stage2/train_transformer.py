import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FILTERED_TRACK_ID_MAP_PKL, PLAYLIST_TRACKS_FILTERED_PARQUET, TRANSFORMER_MODEL_PT
from stage2.dataset_sequence import SequenceDataset
from stage2.models.transformer_model import TransformerModel


DEFAULT_SEQ_LEN = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 8
DEFAULT_MAX_TRAIN_STEPS = 20000
DEFAULT_MAX_EVAL_STEPS = 4000
DEFAULT_EMBED_DIM = 256
DEFAULT_LR = 3e-4
DEFAULT_CANDIDATE_POOL_SIZE = 1024
DEFAULT_TEMPERATURE = 0.07


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage2 transformer as a candidate reranker.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--max-train-steps", type=int, default=DEFAULT_MAX_TRAIN_STEPS)
    parser.add_argument("--max-eval-steps", type=int, default=DEFAULT_MAX_EVAL_STEPS)
    parser.add_argument("--candidate-pool-size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(TRANSFORMER_MODEL_PT))
    return parser.parse_args()


def build_loader(seq_len, batch_size, num_workers, track_map):
    dataset = SequenceDataset(
        seq_len=seq_len,
        db_path=str(PLAYLIST_TRACKS_FILTERED_PARQUET),
        track_map=track_map,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def sample_negative_ids(vocab_size, batch_targets, candidate_pool_size, pad_idx, rng):
    negatives = []
    excluded = set(int(x) for x in batch_targets.tolist())
    excluded.add(int(pad_idx))

    while len(negatives) < candidate_pool_size:
        candidate = int(rng.integers(0, vocab_size))

        if candidate in excluded:
            continue

        negatives.append(candidate)
        excluded.add(candidate)

    return negatives


def evaluate_recall_at_10(model, loader, device, max_eval_steps, candidate_pool_size, pad_idx):
    model.eval()
    correct = 0
    total = 0
    rng = np.random.default_rng(42)

    with torch.no_grad():
        for step, (seq, target) in enumerate(loader):
            seq = seq.to(device)
            target = target.to(device)

            negative_ids = sample_negative_ids(
                vocab_size=model.embedding.num_embeddings - 1,
                batch_targets=target.cpu(),
                candidate_pool_size=candidate_pool_size,
                pad_idx=pad_idx,
                rng=rng,
            )
            negative_tensor = torch.tensor(negative_ids, dtype=torch.long, device=device)
            negative_tensor = negative_tensor.unsqueeze(0).expand(seq.size(0), -1)
            candidate_ids = torch.cat([target.unsqueeze(1), negative_tensor], dim=1)

            scores = model.score_candidates(seq, candidate_ids)
            topk = torch.topk(scores, k=min(10, scores.size(1)), dim=1).indices
            match = (topk == 0).any(dim=1)

            correct += match.sum().item()
            total += target.size(0)

            if step >= max_eval_steps:
                break

    return (correct / total) if total else 0.0


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(
        "Training config:",
        f"epochs={args.epochs}",
        f"batch_size={args.batch_size}",
        f"max_train_steps={args.max_train_steps}",
        f"candidate_pool_size={args.candidate_pool_size}",
        f"temperature={args.temperature}",
    )

    print("Loading track map...")
    with open(FILTERED_TRACK_ID_MAP_PKL, "rb") as f:
        track_map = pickle.load(f)

    vocab_size = len(track_map)
    pad_idx = vocab_size
    print("Vocab size:", vocab_size)

    loader = build_loader(args.seq_len, args.batch_size, args.num_workers, track_map)

    model = TransformerModel(
        vocab_size=vocab_size + 1,
        embed_dim=args.embed_dim,
        max_seq_len=args.seq_len,
        padding_idx=pad_idx,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(42)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        print(f"Epoch {epoch} training...")

        for step, (seq, target) in enumerate(loader):
            seq = seq.to(device)
            target = target.to(device)

            negative_ids = sample_negative_ids(
                vocab_size=vocab_size,
                batch_targets=target.cpu(),
                candidate_pool_size=args.candidate_pool_size,
                pad_idx=pad_idx,
                rng=rng,
            )
            negative_tensor = torch.tensor(negative_ids, dtype=torch.long, device=device)
            negative_tensor = negative_tensor.unsqueeze(0).expand(seq.size(0), -1)
            candidate_ids = torch.cat([target.unsqueeze(1), negative_tensor], dim=1)

            scores = model.score_candidates(seq, candidate_ids) / args.temperature
            labels = torch.zeros(seq.size(0), dtype=torch.long, device=device)
            loss = F.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 500 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.6f}")

            if step >= args.max_train_steps:
                break

        print(f"Epoch {epoch} Total Loss {total_loss:.6f}")

    eval_loader = build_loader(args.seq_len, args.batch_size, args.num_workers, track_map)
    recall_at_10 = evaluate_recall_at_10(
        model=model,
        loader=eval_loader,
        device=device,
        max_eval_steps=args.max_eval_steps,
        candidate_pool_size=args.candidate_pool_size,
        pad_idx=pad_idx,
    )
    print(f"Recall@10: {recall_at_10:.6f}")

    checkpoint_path = Path(args.checkpoint)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size + 1,
            "pad_idx": pad_idx,
            "embed_dim": args.embed_dim,
            "model_kwargs": {
                "vocab_size": vocab_size + 1,
                "embed_dim": args.embed_dim,
                "max_seq_len": args.seq_len,
                "padding_idx": pad_idx,
            },
            "seq_len": args.seq_len,
            "metrics": {
                "recall@10": recall_at_10,
            },
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
