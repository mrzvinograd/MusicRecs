import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import STAGE1_TRACK_VECTORS_NPY
from stage1.candidate_generator import build_stage1_track_vector_cache, load_stage1_assets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute encoded stage1 track vectors for fast recommendation inference."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="How many tracks to encode per model batch.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for the cache. float16 is faster to load and uses less disk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assets = load_stage1_assets()
    assets["track_vectors"] = None
    total = int(assets["embeddings"].shape[0])

    with tqdm(total=total, unit="track") as progress:
        last_done = 0

        def update(done, _total):
            nonlocal last_done
            progress.update(done - last_done)
            last_done = done

        output_path = build_stage1_track_vector_cache(
            assets=assets,
            output_path=STAGE1_TRACK_VECTORS_NPY,
            chunk_size=args.chunk_size,
            dtype=args.dtype,
            progress_callback=update,
        )

    print(f"Stage1 track vector cache saved to: {output_path}")


if __name__ == "__main__":
    main()
