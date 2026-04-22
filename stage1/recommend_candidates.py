import sys
from pathlib import Path
import argparse
import pickle

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import TRACK_EMBEDDINGS_NPY, TRACK2VEC_TRACK_MAP_PKL, TWO_TOWER_MODEL_PT
from stage1.models.two_tower import TwoTowerModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate stage1 candidate tracks for a playlist."
    )
    parser.add_argument(
        "--playlist",
        required=True,
        help="Comma-separated playlist track_rowid values.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="How many candidate tracks to return.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Chunk size when scoring the full embedding matrix.",
    )
    return parser.parse_args()


def build_playlist_tensor(playlist_ids, track_map, embeddings, max_len=20):
    playlist_indices = [track_map[track_id] for track_id in playlist_ids if track_id in track_map]

    if not playlist_indices:
        raise ValueError("None of the provided playlist tracks were found in track_map.pkl")

    playlist_indices = playlist_indices[-max_len:]
    features = embeddings[playlist_indices]

    if len(features) < max_len:
        pad = np.zeros((max_len - len(features), features.shape[1]), dtype=np.float32)
        features = np.vstack([pad, features])

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0), set(playlist_indices)


args = parse_args()

playlist_ids = [int(x.strip()) for x in args.playlist.split(",") if x.strip()]

with open(TRACK2VEC_TRACK_MAP_PKL, "rb") as f:
    track_map = pickle.load(f)

reverse_track_map = {idx: track_id for track_id, idx in track_map.items()}
embeddings = np.load(TRACK_EMBEDDINGS_NPY, mmap_mode="r")

checkpoint = torch.load(TWO_TOWER_MODEL_PT, map_location="cpu")
model_kwargs = checkpoint.get("model_kwargs", {"embed_dim": 512, "hidden_dim": 256})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TwoTowerModel(**model_kwargs).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

playlist_tensor, seen_indices = build_playlist_tensor(
    playlist_ids,
    track_map,
    embeddings,
)
playlist_tensor = playlist_tensor.to(device)

with torch.no_grad():
    playlist_vec = model.encode_playlist(playlist_tensor)

scores = []

for start in range(0, embeddings.shape[0], args.chunk_size):
    stop = min(start + args.chunk_size, embeddings.shape[0])
    chunk = torch.tensor(embeddings[start:stop], dtype=torch.float32, device=device)

    with torch.no_grad():
        chunk_vec = model.encode_tracks(chunk)
        chunk_scores = torch.matmul(playlist_vec, chunk_vec.T).squeeze(0).cpu().numpy()

    scores.append(chunk_scores)

scores = np.concatenate(scores)

for idx in seen_indices:
    scores[idx] = -np.inf

top_indices = np.argsort(-scores)[:args.top_k]

print("Stage1 candidates:")

for rank, idx in enumerate(top_indices, start=1):
    track_id = reverse_track_map[int(idx)]
    print(f"{rank}. track_rowid={track_id} score={float(scores[idx]):.6f}")
