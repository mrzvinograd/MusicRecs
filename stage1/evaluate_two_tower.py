import math

import numpy as np
import torch


def _sample_candidate_indices(dataset, target_idx, candidate_pool_size, rng):
    total_tracks = dataset.embeddings.shape[0]
    sample_size = min(candidate_pool_size - 1, max(0, total_tracks - 1))

    negatives = set()

    while len(negatives) < sample_size:
        candidate = int(rng.integers(0, total_tracks))

        if candidate == target_idx:
            continue

        negatives.add(candidate)

    candidates = [target_idx] + list(negatives)
    rng.shuffle(candidates)
    return candidates


def evaluate_two_tower(
    model,
    dataset,
    k_values=(10, 50),
    num_samples=1000,
    candidate_pool_size=1000,
    seed=42,
):
    device = next(model.parameters()).device
    model.eval()

    rng = np.random.default_rng(seed)
    metrics = {k: {"hits": 0, "ndcg": 0.0, "mrr": 0.0} for k in k_values}
    total = 0

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break

            playlist, _, target_idx = sample
            target_idx = int(target_idx.item())

            candidates = _sample_candidate_indices(
                dataset=dataset,
                target_idx=target_idx,
                candidate_pool_size=candidate_pool_size,
                rng=rng,
            )

            playlist = playlist.unsqueeze(0).to(device)
            candidate_vectors = torch.tensor(
                dataset.get_track_features(candidates),
                dtype=torch.float32,
                device=device,
            )

            playlist_vec = model.encode_playlist(playlist)
            track_vecs = model.encode_tracks(candidate_vectors)
            scores = torch.matmul(playlist_vec, track_vecs.T).squeeze(0)

            target_position = candidates.index(target_idx)

            for k in k_values:
                topk = torch.topk(scores, k=min(k, scores.size(0))).indices.tolist()

                if target_position in topk:
                    metrics[k]["hits"] += 1
                    rank = topk.index(target_position) + 1
                    metrics[k]["ndcg"] += 1.0 / math.log2(rank + 1)
                    metrics[k]["mrr"] += 1.0 / rank

            total += 1

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
