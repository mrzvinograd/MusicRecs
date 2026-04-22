import sys
from pathlib import Path
import argparse
import pickle

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FILTERED_TRACK_ID_MAP_PKL, RANKING_MODEL_PT
from stage1.candidate_generator import generate_stage1_candidates
from stage3.models.model_ranking import RankingModel
from utils.music_metadata import fetch_track_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate final music recommendations using stage1 candidates and stage3 reranking."
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
        help="How many final recommendations to return.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=300,
        help="How many stage1 candidates to pass into the reranker.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Chunk size for stage1 full-catalog scoring.",
    )
    return parser.parse_args()


def build_stage3_playlist_indices(playlist_track_ids, filtered_track_map, pad_idx, max_len=20):
    playlist_indices = [filtered_track_map[track_id] for track_id in playlist_track_ids if track_id in filtered_track_map]

    if not playlist_indices:
        raise ValueError("None of the provided playlist tracks were found in the stage3 filtered track map.")

    playlist_indices = playlist_indices[-max_len:]

    if len(playlist_indices) < max_len:
        playlist_indices = [pad_idx] * (max_len - len(playlist_indices)) + playlist_indices

    return playlist_indices


def rerank_candidates(model, playlist_track_ids, candidate_track_ids, filtered_track_map, pad_idx, device):
    playlist_indices = build_stage3_playlist_indices(
        playlist_track_ids=playlist_track_ids,
        filtered_track_map=filtered_track_map,
        pad_idx=pad_idx,
    )
    candidate_indices = [filtered_track_map[track_id] for track_id in candidate_track_ids if track_id in filtered_track_map]

    if not candidate_indices:
        raise ValueError("No stage1 candidates were available in the stage3 filtered track map.")

    playlist_tensor = torch.tensor([playlist_indices], dtype=torch.long, device=device)
    candidate_tensor = torch.tensor([candidate_indices], dtype=torch.long, device=device)

    with torch.no_grad():
        scores = model(playlist_tensor, candidate_tensor).squeeze(0).cpu().numpy()

    reverse_filtered_map = {idx: track_id for track_id, idx in filtered_track_map.items()}

    ranked = np.argsort(-scores)
    return [
        {
            "track_id": reverse_filtered_map[candidate_indices[pos]],
            "score": float(scores[pos]),
        }
        for pos in ranked
    ]


args = parse_args()

playlist_track_ids = [int(x.strip()) for x in args.playlist.split(",") if x.strip()]

stage1_candidates = generate_stage1_candidates(
    playlist_ids=playlist_track_ids,
    top_k=args.candidate_k,
    chunk_size=args.chunk_size,
)

candidate_track_ids = [item["track_id"] for item in stage1_candidates]

with open(FILTERED_TRACK_ID_MAP_PKL, "rb") as f:
    filtered_track_map = pickle.load(f)

checkpoint = torch.load(RANKING_MODEL_PT, map_location="cpu")
model_kwargs = checkpoint.get(
    "model_kwargs",
    {
        "vocab_size": checkpoint["vocab_size"],
        "padding_idx": checkpoint["pad_idx"],
    },
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RankingModel(**model_kwargs).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
pad_idx = model_kwargs["padding_idx"]

ranked_candidates = rerank_candidates(
    model=model,
    playlist_track_ids=playlist_track_ids,
    candidate_track_ids=candidate_track_ids,
    filtered_track_map=filtered_track_map,
    pad_idx=pad_idx,
    device=device,
)

final_items = ranked_candidates[:args.top_k]
metadata = fetch_track_metadata([item["track_id"] for item in final_items])

print("Final recommendations:")

for rank, item in enumerate(final_items, start=1):
    meta = metadata.get(item["track_id"], {})
    track_name = meta.get("track_name", "Unknown Track")
    artist_names = meta.get("artist_names", "Unknown Artist")
    album_name = meta.get("album_name", "Unknown Album")

    print(
        f"{rank}. track_rowid={item['track_id']} "
        f"score={item['score']:.6f} "
        f"| {track_name} - {artist_names} "
        f"| album={album_name}"
    )
