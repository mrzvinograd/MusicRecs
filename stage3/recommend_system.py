import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import TRANSFORMER_MODEL_PT
from stage1.candidate_generator import generate_stage1_candidates, load_stage1_assets
from stage2.model_infer import score_sequence_model
from stage2.pipeline import load_stage2_assets
from stage3.pipeline import load_stage3_assets, rerank_candidates
from utils.music_metadata import fetch_track_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end music recommendation: playlist track ids in, stage outputs and final recommendations out."
    )
    parser.add_argument(
        "--playlist",
        required=True,
        help="Comma-separated playlist track ids (Spotify/local dataset track_rowid values).",
    )
    parser.add_argument("--top-k", type=int, default=20, help="How many final recommendations to return.")
    parser.add_argument("--candidate-k", type=int, default=300, help="How many stage1 candidates to generate.")
    parser.add_argument(
        "--stage2-k",
        type=int,
        default=100,
        help="How many top candidates to keep after stage2 sequence scoring before stage3 reranking.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of human-friendly text.",
    )
    return parser.parse_args()


def enrich_items(track_ids, score_lookup):
    metadata = fetch_track_metadata(track_ids)
    items = []

    for rank, track_id in enumerate(track_ids, start=1):
        meta = metadata.get(track_id, {})
        items.append(
            {
                "rank": rank,
                "track_id": track_id,
                "score": float(score_lookup.get(track_id, 0.0)),
                "track_name": meta.get("track_name", "Unknown Track"),
                "artist_names": meta.get("artist_names", "Unknown Artist"),
                "album_name": meta.get("album_name", "Unknown Album"),
            }
        )

    return items


def print_section(title, items):
    print(title)
    if not items:
        print("  no items")
        return

    for item in items:
        print(
            f"  {item['rank']}. track_id={item['track_id']} score={item['score']:.6f} "
            f"| {item['track_name']} - {item['artist_names']} | album={item['album_name']}"
        )


def main():
    args = parse_args()
    playlist_track_ids = [int(x.strip()) for x in args.playlist.split(",") if x.strip()]

    if not playlist_track_ids:
        raise ValueError("Playlist is empty. Provide at least one track id.")

    stage1_assets = load_stage1_assets()
    stage3_assets = load_stage3_assets()

    stage2_assets = None
    stage2_available = TRANSFORMER_MODEL_PT.exists()
    if stage2_available:
        try:
            stage2_assets = load_stage2_assets()
        except Exception as exc:
            stage2_available = False
            stage2_error = str(exc)
        else:
            stage2_error = None
    else:
        stage2_error = f"Missing checkpoint: {TRANSFORMER_MODEL_PT}"

    stage1_candidates = generate_stage1_candidates(
        playlist_ids=playlist_track_ids,
        top_k=args.candidate_k,
        assets=stage1_assets,
    )
    stage1_score_lookup = {item["track_id"]: item["score"] for item in stage1_candidates}
    stage1_ids = [item["track_id"] for item in stage1_candidates]

    if stage2_available and stage2_assets is not None:
        stage2_scores = score_sequence_model(
            model=stage2_assets["model"],
            playlist_track_ids=playlist_track_ids,
            candidate_track_ids=stage1_ids,
            track_map=stage2_assets["track_map"],
            pad_idx=stage2_assets["pad_idx"],
            device=stage2_assets["device"],
        )
        stage2_ids = sorted(stage1_ids, key=lambda track_id: stage2_scores.get(track_id, 0.0), reverse=True)[: args.stage2_k]
    else:
        stage2_scores = {}
        stage2_ids = stage1_ids[: args.stage2_k]

    final_ranked = rerank_candidates(
        model=stage3_assets["model"],
        playlist_track_ids=playlist_track_ids,
        candidate_track_ids=stage2_ids,
        filtered_track_map=stage3_assets["track_map"],
        pad_idx=stage3_assets["pad_idx"],
        device=stage3_assets["device"],
    )
    final_ids = [item["track_id"] for item in final_ranked[: args.top_k]]
    final_score_lookup = {item["track_id"]: item["score"] for item in final_ranked}

    result = {
        "input_playlist": playlist_track_ids,
        "stage2_enabled": stage2_available,
        "stage2_error": stage2_error,
        "stage1_top": enrich_items(stage1_ids[: args.top_k], stage1_score_lookup),
        "stage2_top": enrich_items(stage2_ids[: args.top_k], stage2_scores if stage2_available else stage1_score_lookup),
        "final_top": enrich_items(final_ids, final_score_lookup),
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"Input playlist: {playlist_track_ids}")
    print(f"Stage2 enabled: {'yes' if stage2_available else 'no'}")
    if stage2_error:
        print(f"Stage2 note: {stage2_error}")
    print_section("\nStage1 candidates:", result["stage1_top"])
    print_section("\nStage2 reranked candidates:", result["stage2_top"])
    print_section("\nFinal recommendations:", result["final_top"])


if __name__ == "__main__":
    main()
