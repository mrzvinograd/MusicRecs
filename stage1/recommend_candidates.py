import sys
from pathlib import Path
import argparse


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stage1.candidate_generator import generate_stage1_candidates


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

args = parse_args()

playlist_ids = [int(x.strip()) for x in args.playlist.split(",") if x.strip()]
candidates = generate_stage1_candidates(
    playlist_ids=playlist_ids,
    top_k=args.top_k,
    chunk_size=args.chunk_size,
)

print("Stage1 candidates:")

for item in candidates:
    print(f"{item['rank']}. track_rowid={item['track_id']} score={item['score']:.6f}")
