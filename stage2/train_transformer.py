import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FILTERED_TRACK_ID_MAP_PKL, PLAYLIST_TRACKS_PARQUET, TRANSFORMER_MODEL_PT
from stage2.dataset_sequence import SequenceDataset
from stage2.models.transformer_model import TransformerModel


DEFAULT_SEQ_LEN = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 8
DEFAULT_MAX_TRAIN_STEPS = 20000
DEFAULT_MAX_EVAL_STEPS = 4000
DEFAULT_EMBED_DIM = 256


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage2 transformer next-track model.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--max-train-steps", type=int, default=DEFAULT_MAX_TRAIN_STEPS)
    parser.add_argument("--max-eval-steps", type=int, default=DEFAULT_MAX_EVAL_STEPS)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(TRANSFORMER_MODEL_PT))
    return parser.parse_args()


def build_loader(seq_len, batch_size, num_workers, track_map):
    dataset = SequenceDataset(
        seq_len=seq_len,
        db_path=str(PLAYLIST_TRACKS_PARQUET),
        track_map=track_map,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def evaluate_recall_at_10(model, loader, device, max_eval_steps):
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

            if step >= max_eval_steps:
                break

    return (correct / total) if total else 0.0


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading track map...")
    with open(FILTERED_TRACK_ID_MAP_PKL, "rb") as f:
        track_map = pickle.load(f)

    vocab_size = len(track_map)
    pad_idx = vocab_size
    print("Vocab size:", vocab_size)

    loader = build_loader(args.seq_len, args.batch_size, args.num_workers, track_map)

    model = TransformerModel(vocab_size=vocab_size + 1, embed_dim=args.embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        print(f"Epoch {epoch} training...")

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
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.6f}")

            if step >= args.max_train_steps:
                break

        print(f"Epoch {epoch} Total Loss {total_loss:.6f}")

    eval_loader = build_loader(args.seq_len, args.batch_size, args.num_workers, track_map)
    recall_at_10 = evaluate_recall_at_10(model, eval_loader, device, args.max_eval_steps)
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
