import math
import sys
from pathlib import Path

import duckdb


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PLAYLIST_TRACKS_PARQUET
from stage1.candidate_generator import generate_stage1_candidates, load_stage1_assets
from stage3.pipeline import load_stage3_assets, rerank_candidates


DB_PATH = str(PLAYLIST_TRACKS_PARQUET)
EVAL_MOD = 20
EVAL_REMAINDER = 0
NUM_SAMPLES = 200
STAGE1_CANDIDATE_K = 300
K_VALUES = (10, 50)


def stream_eval_playlists(db_path, stage1_track_map, eval_mod=20, eval_remainder=0, max_samples=200):
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='16GB'")

    query = f"""
    SELECT playlist_rowid, track_rowid
    FROM read_parquet('{db_path}')
    WHERE track_rowid IS NOT NULL
    ORDER BY playlist_rowid, position
    """

    cursor = con.execute(query)

    current_playlist = []
    last_pid = None
    yielded = 0

    while True:
        batch = cursor.fetchmany(100000)

        if not batch:
            break

        for pid, track in batch:
            if track not in stage1_track_map:
                continue

            if last_pid is not None and pid != last_pid:
                if len(current_playlist) > 1 and last_pid % eval_mod == eval_remainder:
                    yield current_playlist[:-1], current_playlist[-1]
                    yielded += 1

                    if yielded >= max_samples:
                        return

                current_playlist = []

            current_playlist.append(track)
            last_pid = pid

    if len(current_playlist) > 1 and last_pid % eval_mod == eval_remainder and yielded < max_samples:
        yield current_playlist[:-1], current_playlist[-1]


def update_metrics(metrics, ranked_track_ids, target_track_id, k_values):
    for k in k_values:
        topk = ranked_track_ids[:k]

        if target_track_id not in topk:
            continue

        metrics[k]["hits"] += 1
        rank = topk.index(target_track_id) + 1
        metrics[k]["ndcg"] += 1.0 / math.log2(rank + 1)
        metrics[k]["mrr"] += 1.0 / rank


stage1_assets = load_stage1_assets()
stage3_assets = load_stage3_assets()

stage1_metrics = {k: {"hits": 0, "ndcg": 0.0, "mrr": 0.0} for k in K_VALUES}
final_metrics = {k: {"hits": 0, "ndcg": 0.0, "mrr": 0.0} for k in K_VALUES}
total = 0

for playlist_track_ids, target_track_id in stream_eval_playlists(
    db_path=DB_PATH,
    stage1_track_map=stage1_assets["track_map"],
    eval_mod=EVAL_MOD,
    eval_remainder=EVAL_REMAINDER,
    max_samples=NUM_SAMPLES,
):
    stage1_candidates = generate_stage1_candidates(
        playlist_ids=playlist_track_ids,
        top_k=STAGE1_CANDIDATE_K,
        assets=stage1_assets,
    )

    stage1_candidate_ids = [item["track_id"] for item in stage1_candidates]
    update_metrics(stage1_metrics, stage1_candidate_ids, target_track_id, K_VALUES)

    final_ranked = rerank_candidates(
        model=stage3_assets["model"],
        playlist_track_ids=playlist_track_ids,
        candidate_track_ids=stage1_candidate_ids,
        filtered_track_map=stage3_assets["track_map"],
        pad_idx=stage3_assets["pad_idx"],
        device=stage3_assets["device"],
    )

    final_ranked_ids = [item["track_id"] for item in final_ranked]
    update_metrics(final_metrics, final_ranked_ids, target_track_id, K_VALUES)
    total += 1


def print_metrics(title, metrics, total):
    print(title)

    if total == 0:
        print("No evaluation samples.")
        return

    for k in K_VALUES:
        print(
            f"recall@{k}: {metrics[k]['hits'] / total:.4f} "
            f"ndcg@{k}: {metrics[k]['ndcg'] / total:.4f} "
            f"mrr@{k}: {metrics[k]['mrr'] / total:.4f}"
        )


print(f"Evaluated samples: {total}")
print_metrics("Stage1 candidate metrics", stage1_metrics, total)
print_metrics("Final pipeline metrics", final_metrics, total)
