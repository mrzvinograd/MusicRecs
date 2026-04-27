import torch


def score_sequence_model(
    model,
    playlist_track_ids,
    candidate_track_ids,
    track_map,
    pad_idx,
    device,
    max_len=20,
):
    model.eval()

    # переводим playlist в индексы
    indices = [track_map[t] for t in playlist_track_ids if t in track_map]

    if len(indices) == 0:
        return {}

    indices = indices[-max_len:]

    padded = [pad_idx] * (max_len - len(indices)) + indices
    x = torch.tensor([padded], dtype=torch.long).to(device)
    candidate_indices = [track_map[track_id] for track_id in candidate_track_ids if track_id in track_map]

    if not candidate_indices:
        return {track_id: 0.0 for track_id in candidate_track_ids}

    candidate_tensor = torch.tensor([candidate_indices], dtype=torch.long, device=device)

    with torch.no_grad():
        scores_tensor = model.score_candidates(x, candidate_tensor).squeeze(0).cpu().tolist()

    scores = {}
    score_by_index = {
        idx: float(score)
        for idx, score in zip(candidate_indices, scores_tensor)
    }

    for track_id in candidate_track_ids:
        if track_id in track_map:
            idx = track_map[track_id]
            scores[track_id] = score_by_index.get(idx, 0.0)
        else:
            scores[track_id] = 0.0

    return scores
