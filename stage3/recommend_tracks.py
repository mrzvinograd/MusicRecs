import sys
from pathlib import Path

import argparse
import pickle

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    RANKING_MODEL_PT,
    TRACK_EMBEDDINGS_NPY,
    TRACK_IDS_TXT,
    TRACK_ID_MAP_PKL,
)
from stage3.models.model_ranking import RankingModel


CHECKPOINT_PATH = RANKING_MODEL_PT
TRACK_MAP_PATH = TRACK_ID_MAP_PKL
EMBEDDING_FILE = TRACK_EMBEDDINGS_NPY
EMBEDDING_IDS_FILE = TRACK_IDS_TXT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend tracks for a playlist using the ranking model."
    )
    parser.add_argument(
        "--playlist",
        required=True,
        help="Comma-separated playlist track_rowid values.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many recommendations to return.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=1000,
        help="How many candidate tracks to score before taking top-k.",
    )
    parser.add_argument(
        "--neighbors-per-track",
        type=int,
        default=150,
        help="How many nearest neighbors to pull from track2vec for each playlist track.",
    )
    return parser.parse_args()


def load_embedding_index():
    embeddings = np.load(EMBEDDING_FILE, mmap_mode="r")

    with open(EMBEDDING_IDS_FILE, "r", encoding="utf-8") as f:
        track_ids = [int(line.strip()) for line in f if line.strip()]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / norms

    embedding_index_by_track = {
        track_id: idx for idx, track_id in enumerate(track_ids)
    }

    return normalized, track_ids, embedding_index_by_track


def build_candidates(
    playlist_track_ids,
    track_map,
    normalized_embeddings,
    embedding_track_ids,
    embedding_index_by_track,
    neighbors_per_track,
    candidate_k,
):
    seen = set(playlist_track_ids)
    candidate_scores = {}

    recent_tracks = playlist_track_ids[-5:]

    for weight_rank, track_id in enumerate(reversed(recent_tracks), start=1):
        embedding_idx = embedding_index_by_track.get(track_id)

        if embedding_idx is None:
            continue

        query = normalized_embeddings[embedding_idx]
        scores = normalized_embeddings @ query
        top_n = min(neighbors_per_track + 1, scores.shape[0])
        top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
        ordered = top_idx[np.argsort(-scores[top_idx])]

        recency_weight = 1.0 / weight_rank

        for idx in ordered:
            raw_track_id = int(embedding_track_ids[idx])

            if raw_track_id in seen:
                continue

            if raw_track_id not in track_map:
                continue

            mapped_idx = track_map[raw_track_id]
            candidate_scores[mapped_idx] = max(
                candidate_scores.get(mapped_idx, float("-inf")),
                float(scores[idx]) * recency_weight,
            )

    if len(candidate_scores) < candidate_k:
        for raw_track_id, mapped_idx in track_map.items():
            if raw_track_id in seen or mapped_idx in candidate_scores:
                continue

            candidate_scores[mapped_idx] = -1e9

            if len(candidate_scores) >= candidate_k:
                break

    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    return [mapped_idx for mapped_idx, _ in ranked[:candidate_k]]


def score_candidates(model, playlist_indices, candidate_indices, pad_idx, device):
    playlist_tensor = torch.tensor([playlist_indices], dtype=torch.long, device=device)
    candidate_tensor = torch.tensor([candidate_indices], dtype=torch.long, device=device)

    with torch.no_grad():
        scores = model(playlist_tensor, candidate_tensor).squeeze(0)

    return scores.cpu().numpy()


args = parse_args()

playlist_track_ids = [int(x.strip()) for x in args.playlist.split(",") if x.strip()]

with open(TRACK_MAP_PATH, "rb") as f:
    track_map = pickle.load(f)

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model_kwargs = checkpoint.get(
    "model_kwargs",
    {
        "vocab_size": checkpoint["vocab_size"],
        "padding_idx": checkpoint["pad_idx"],
    },
)
pad_idx = model_kwargs["padding_idx"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RankingModel(**model_kwargs).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

playlist_indices = [track_map[track_id] for track_id in playlist_track_ids if track_id in track_map]

if not playlist_indices:
    raise ValueError("None of the provided playlist tracks were found in track_map.pkl")

if len(playlist_indices) > 20:
    playlist_indices = playlist_indices[-20:]

normalized_embeddings, embedding_track_ids, embedding_index_by_track = load_embedding_index()

candidates = build_candidates(
    playlist_track_ids=playlist_track_ids,
    track_map=track_map,
    normalized_embeddings=normalized_embeddings,
    embedding_track_ids=embedding_track_ids,
    embedding_index_by_track=embedding_index_by_track,
    neighbors_per_track=args.neighbors_per_track,
    candidate_k=args.candidate_k,
)

if not candidates:
    raise ValueError("No candidates were generated for the provided playlist.")

scores = score_candidates(
    model=model,
    playlist_indices=playlist_indices,
    candidate_indices=candidates,
    pad_idx=pad_idx,
    device=device,
)

reverse_track_map = {idx: track_id for track_id, idx in track_map.items()}
top_order = np.argsort(-scores)[:args.top_k]

print("Recommendations:")

for rank, pos in enumerate(top_order, start=1):
    mapped_idx = candidates[int(pos)]
    track_id = reverse_track_map[mapped_idx]
    print(f"{rank}. track_rowid={track_id} score={scores[int(pos)]:.6f}")
