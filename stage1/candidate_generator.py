import pickle

import numpy as np
import torch

from config import TRACK_EMBEDDINGS_NPY, TRACK2VEC_TRACK_MAP_PKL, TWO_TOWER_MODEL_PT
from stage1.models.two_tower import TwoTowerModel


def load_stage1_assets(
    checkpoint_path=TWO_TOWER_MODEL_PT,
    embeddings_path=TRACK_EMBEDDINGS_NPY,
    track_map_path=TRACK2VEC_TRACK_MAP_PKL,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(track_map_path, "rb") as f:
        track_map = pickle.load(f)

    reverse_track_map = {idx: track_id for track_id, idx in track_map.items()}
    embeddings = np.load(embeddings_path, mmap_mode="r")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_kwargs = checkpoint.get("model_kwargs", {"embed_dim": 512, "hidden_dim": 256})
    else:
        # Backward compatibility with older checkpoints saved as raw state_dict
        state_dict = checkpoint
        model_kwargs = {"embed_dim": 512, "hidden_dim": 256}

    model = TwoTowerModel(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "device": device,
        "model": model,
        "embeddings": embeddings,
        "track_map": track_map,
        "reverse_track_map": reverse_track_map,
    }


def build_playlist_tensor(playlist_ids, track_map, embeddings, max_len=20):
    playlist_indices = [track_map[track_id] for track_id in playlist_ids if track_id in track_map]

    if not playlist_indices:
        raise ValueError("None of the provided playlist tracks were found in the stage1 track map.")

    playlist_indices = playlist_indices[-max_len:]
    features = embeddings[playlist_indices]

    if len(features) < max_len:
        pad = np.zeros((max_len - len(features), features.shape[1]), dtype=np.float32)
        features = np.vstack([pad, features])

    playlist_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return playlist_tensor, set(playlist_indices)


def generate_stage1_candidates(
    playlist_ids,
    top_k=100,
    chunk_size=4096,
    max_len=20,
    assets=None,
):
    if assets is None:
        assets = load_stage1_assets()

    model = assets["model"]
    device = assets["device"]
    embeddings = assets["embeddings"]
    track_map = assets["track_map"]
    reverse_track_map = assets["reverse_track_map"]

    playlist_tensor, seen_indices = build_playlist_tensor(
        playlist_ids=playlist_ids,
        track_map=track_map,
        embeddings=embeddings,
        max_len=max_len,
    )
    playlist_tensor = playlist_tensor.to(device)

    with torch.no_grad():
        playlist_vec = model.encode_playlist(playlist_tensor)

    scores = []

    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])
        chunk = torch.tensor(embeddings[start:stop], dtype=torch.float32, device=device)

        with torch.no_grad():
            chunk_vec = model.encode_tracks(chunk)
            chunk_scores = torch.matmul(playlist_vec, chunk_vec.T).squeeze(0).cpu().numpy()

        scores.append(chunk_scores)

    scores = np.concatenate(scores)

    for idx in seen_indices:
        scores[idx] = -np.inf

    top_indices = np.argsort(-scores)[:top_k]

    return [
        {
            "rank": rank,
            "track_id": reverse_track_map[int(idx)],
            "mapped_idx": int(idx),
            "score": float(scores[idx]),
        }
        for rank, idx in enumerate(top_indices, start=1)
    ]

