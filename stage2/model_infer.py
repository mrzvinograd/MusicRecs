import torch


def score_sequence_model(
    model,
    playlist_track_ids,
    candidate_track_ids,
    track_map,
    pad_idx,
    device,
    max_len=50,
):
    model.eval()

    # переводим playlist в индексы
    indices = [track_map[t] for t in playlist_track_ids if t in track_map]

    if len(indices) == 0:
        return {}

    indices = indices[-max_len:]

    padded = [pad_idx] * (max_len - len(indices)) + indices
    x = torch.tensor([padded], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)  # shape [1, vocab_size]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    scores = {}

    for track_id in candidate_track_ids:
        if track_id in track_map:
            idx = track_map[track_id]
            scores[track_id] = float(probs[idx])
        else:
            scores[track_id] = 0.0

    return scores
