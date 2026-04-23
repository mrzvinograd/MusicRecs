import sys
from pathlib import Path
import argparse


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stage1.candidate_generator import generate_stage1_candidates
from stage3.pipeline import load_stage3_assets, rerank_candidates
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

args = parse_args()

playlist_track_ids = [int(x.strip()) for x in args.playlist.split(",") if x.strip()]

stage1_candidates = generate_stage1_candidates(
    playlist_ids=playlist_track_ids,
    top_k=args.candidate_k,
    chunk_size=args.chunk_size,
)

candidate_track_ids = [item["track_id"] for item in stage1_candidates]

stage3_assets = load_stage3_assets()

ranked_candidates = rerank_candidates(
    model=stage3_assets["model"],
    playlist_track_ids=playlist_track_ids,
    candidate_track_ids=candidate_track_ids,
    filtered_track_map=stage3_assets["track_map"],
    pad_idx=stage3_assets["pad_idx"],
    device=stage3_assets["device"],
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
