import argparse
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PLAYLIST_TRACKS_PARQUET, TRANSFORMER_MODEL_PT
from stage1.candidate_generator import generate_stage1_candidates, load_stage1_assets
from stage2.model_infer import score_sequence_model
from stage2.pipeline import load_stage2_assets
from stage3.eval_pipeline import stream_eval_playlists
from stage3.pipeline import load_stage3_assets, rerank_candidates


DEFAULT_K_VALUES = (10, 50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate recall/NDCG/MRR for stage1, stage2, and full pipeline on holdout playlists."
    )
    parser.add_argument("--samples", type=int, default=50, help="How many evaluation playlists to use.")
    parser.add_argument("--candidate-k", type=int, default=300, help="How many stage1 candidates to generate.")
    parser.add_argument("--stage2-k", type=int, default=100, help="How many candidates to keep after stage2.")
    parser.add_argument("--k-values", default="10,50", help="Comma-separated K values, for example 10,50.")
    return parser.parse_args()


def update_metrics(metrics, ranked_track_ids, target_track_id, k_values):
    for k in k_values:
        topk = ranked_track_ids[:k]
        if target_track_id not in topk:
            continue
        metrics[k]["hits"] += 1
        rank = topk.index(target_track_id) + 1
        metrics[k]["ndcg"] += 1.0 / math.log2(rank + 1)
        metrics[k]["mrr"] += 1.0 / rank


def format_metrics(metrics, total, k_values):
    lines = []
    for k in k_values:
        recall = metrics[k]["hits"] / total if total else 0.0
        ndcg = metrics[k]["ndcg"] / total if total else 0.0
        mrr = metrics[k]["mrr"] / total if total else 0.0
        lines.append(f"  @k={k}: recall={recall:.4f} ndcg={ndcg:.4f} mrr={mrr:.4f}")
    return "\n".join(lines)


def main():
    args = parse_args()
    k_values = tuple(int(x.strip()) for x in args.k_values.split(",") if x.strip()) or DEFAULT_K_VALUES

    stage1_assets = load_stage1_assets()
    stage3_assets = load_stage3_assets()

    stage2_assets = None
    stage2_available = TRANSFORMER_MODEL_PT.exists()
    if stage2_available:
        try:
            stage2_assets = load_stage2_assets()
        except Exception:
            stage2_available = False

    stage1_metrics = {k: {"hits": 0, "ndcg": 0.0, "mrr": 0.0} for k in k_values}
    stage2_metrics = {k: {"hits": 0, "ndcg": 0.0, "mrr": 0.0} for k in k_values}
    final_metrics = {k: {"hits": 0, "ndcg": 0.0, "mrr": 0.0} for k in k_values}
    total = 0

    for playlist_track_ids, target_track_id in stream_eval_playlists(
        db_path=str(PLAYLIST_TRACKS_PARQUET),
        stage1_track_map=stage1_assets["track_map"],
        max_samples=args.samples,
    ):
        stage1_candidates = generate_stage1_candidates(
            playlist_ids=playlist_track_ids,
            top_k=args.candidate_k,
            assets=stage1_assets,
        )
        stage1_ids = [item["track_id"] for item in stage1_candidates]
        update_metrics(stage1_metrics, stage1_ids, target_track_id, k_values)

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
            stage2_ids = stage1_ids[: args.stage2_k]

        update_metrics(stage2_metrics, stage2_ids, target_track_id, k_values)

        final_ranked = rerank_candidates(
            model=stage3_assets["model"],
            playlist_track_ids=playlist_track_ids,
            candidate_track_ids=stage2_ids,
            filtered_track_map=stage3_assets["track_map"],
            pad_idx=stage3_assets["pad_idx"],
            device=stage3_assets["device"],
        )
        final_ids = [item["track_id"] for item in final_ranked]
        update_metrics(final_metrics, final_ids, target_track_id, k_values)
        total += 1

    print(f"Evaluated playlists: {total}")
    print(f"Stage2 enabled: {'yes' if stage2_available else 'no'}")
    print("\nStage1 metrics:")
    print(format_metrics(stage1_metrics, total, k_values))
    print("\nStage2 metrics:")
    print(format_metrics(stage2_metrics, total, k_values))
    print("\nFull pipeline metrics:")
    print(format_metrics(final_metrics, total, k_values))


if __name__ == "__main__":
    main()
