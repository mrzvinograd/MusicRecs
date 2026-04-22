import math

import torch
from torch.utils.data import DataLoader

from stage3.dataset_ranking import RankingEvalDataset, ranking_collate


def evaluate_ranking(
    model,
    db_path,
    track_map,
    pad_idx,
    k_values=(10,),
    num_candidates=100,
    num_samples=1000,
    batch_size=32,
    eval_mod=20,
    eval_remainder=0,
    use_hard_negatives=True,
    hard_negative_ratio=0.7,
    hard_pool_size=128,
):
    device = next(model.parameters()).device
    model.eval()

    dataset = RankingEvalDataset(
        db_path=db_path,
        track_map=track_map,
        num_candidates=num_candidates,
        max_samples=num_samples,
        eval_mod=eval_mod,
        eval_remainder=eval_remainder,
        use_hard_negatives=use_hard_negatives,
        hard_negative_ratio=hard_negative_ratio,
        hard_pool_size=hard_pool_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=lambda batch: ranking_collate(batch, pad_idx),
    )

    metrics = {k: {"hits": 0, "ndcg": 0.0, "mrr": 0.0} for k in k_values}
    total = 0

    with torch.no_grad():
        for playlist, candidates, target_idx in loader:
            playlist = playlist.to(device)
            candidates = candidates.to(device)
            target_idx = target_idx.to(device)

            scores = model(playlist, candidates)

            for k in k_values:
                topk = torch.topk(scores, k=min(k, scores.size(1)), dim=1).indices
                matches = topk.eq(target_idx.unsqueeze(1))

                metrics[k]["hits"] += matches.any(dim=1).sum().item()

                for row in range(topk.size(0)):
                    match_pos = torch.nonzero(matches[row], as_tuple=False)

                    if match_pos.numel() == 0:
                        continue

                    rank = match_pos[0].item() + 1
                    metrics[k]["ndcg"] += 1.0 / math.log2(rank + 1)
                    metrics[k]["mrr"] += 1.0 / rank

            total += target_idx.size(0)

    if total == 0:
        return {
            f"recall@{k}": 0.0 for k in k_values
        } | {
            f"ndcg@{k}": 0.0 for k in k_values
        } | {
            f"mrr@{k}": 0.0 for k in k_values
        }

    results = {}

    for k in k_values:
        results[f"recall@{k}"] = metrics[k]["hits"] / total
        results[f"ndcg@{k}"] = metrics[k]["ndcg"] / total
        results[f"mrr@{k}"] = metrics[k]["mrr"] / total

    return results
