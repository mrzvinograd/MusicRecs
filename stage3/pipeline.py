import pickle

import torch

from config import FILTERED_TRACK_ID_MAP_PKL, RANKING_MODEL_PT
from stage3.models.model_ranking import RankingModel


def load_stage3_assets(
    checkpoint_path=RANKING_MODEL_PT,
    track_map_path=FILTERED_TRACK_ID_MAP_PKL,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(track_map_path, "rb") as f:
        track_map = pickle.load(f)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_kwargs = checkpoint.get(
            "model_kwargs",
            {
                "vocab_size": checkpoint.get("vocab_size", len(track_map) + 1),
                "padding_idx": checkpoint.get("pad_idx", len(track_map)),
            },
        )
        pad_idx = checkpoint.get("pad_idx", model_kwargs.get("padding_idx", len(track_map)))
    else:
        state_dict = checkpoint
        pad_idx = len(track_map)
        model_kwargs = {
            "vocab_size": len(track_map) + 1,
            "padding_idx": pad_idx,
        }

    model = RankingModel(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "device": device,
        "model": model,
        "track_map": track_map,
        "pad_idx": pad_idx,
    }


def build_stage3_playlist_indices(playlist_track_ids, filtered_track_map, max_len=20):
    playlist_indices = [filtered_track_map[track_id] for track_id in playlist_track_ids if track_id in filtered_track_map]
    return playlist_indices[-max_len:]


def rerank_candidates(
    model,
    playlist_track_ids,
    candidate_track_ids,
    filtered_track_map,
    pad_idx,
    device,
    max_len=20,
):
    model.eval()

    playlist_indices = build_stage3_playlist_indices(
        playlist_track_ids=playlist_track_ids,
        filtered_track_map=filtered_track_map,
        max_len=max_len,
    )

    if not playlist_indices:
        return []

    candidate_pairs = [
        (track_id, filtered_track_map[track_id])
        for track_id in candidate_track_ids
        if track_id in filtered_track_map
    ]

    if not candidate_pairs:
        return []

    playlist_tensor = torch.tensor([playlist_indices], dtype=torch.long)

    if playlist_tensor.size(1) < max_len:
        pad = torch.full((1, max_len - playlist_tensor.size(1)), pad_idx, dtype=torch.long)
        playlist_tensor = torch.cat([pad, playlist_tensor], dim=1)

    candidate_tensor = torch.tensor([[mapped_idx for _, mapped_idx in candidate_pairs]], dtype=torch.long)

    playlist_tensor = playlist_tensor.to(device)
    candidate_tensor = candidate_tensor.to(device)

    with torch.no_grad():
        scores = model(playlist_tensor, candidate_tensor).squeeze(0).cpu().tolist()

    ranked = sorted(
        zip([track_id for track_id, _ in candidate_pairs], scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [{"track_id": track_id, "score": float(score)} for track_id, score in ranked]
