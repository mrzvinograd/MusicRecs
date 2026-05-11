import pickle

import numpy as np
import torch

from config import STAGE1_TRACK_VECTORS_NPY, TRACK_EMBEDDINGS_NPY, TRACK2VEC_TRACK_MAP_PKL, TWO_TOWER_MODEL_PT
from stage1.models.two_tower import TwoTowerModel
from utils.audio_features import load_aligned_audio_features, load_audio_feature_lookup


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
    reverse_track_ids = np.empty(len(track_map), dtype=np.int64)
    for track_id, mapped_idx in track_map.items():
        reverse_track_ids[mapped_idx] = track_id

    embeddings = np.load(embeddings_path, mmap_mode="r")
    audio_features = load_audio_feature_lookup()

    if audio_features is None:
        audio_features = load_aligned_audio_features(track_map)

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
        "audio_features": audio_features,
        "track_map": track_map,
        "reverse_track_map": reverse_track_map,
        "reverse_track_ids": reverse_track_ids,
        "track_vectors": _load_track_vector_cache(
            STAGE1_TRACK_VECTORS_NPY,
            embeddings.shape[0],
            dependency_paths=(checkpoint_path, embeddings_path),
        ),
    }


def _load_track_vector_cache(cache_path, expected_rows, dependency_paths=()):
    if not cache_path.exists():
        return None

    cache_mtime = cache_path.stat().st_mtime
    for dependency_path in dependency_paths:
        if dependency_path.exists() and dependency_path.stat().st_mtime > cache_mtime:
            return None

    track_vectors = np.load(cache_path, mmap_mode="r")

    if track_vectors.ndim != 2 or track_vectors.shape[0] != expected_rows:
        return None

    return track_vectors


def _combine_track_features(indices, embeddings, audio_features, reverse_track_ids=None):
    base = np.asarray(embeddings[indices], dtype=np.float32)

    if audio_features is None:
        return base

    if hasattr(audio_features, "features_for_indices"):
        if reverse_track_ids is None:
            raise ValueError("reverse_track_ids is required for lazy audio feature lookup.")
        extra = audio_features.features_for_indices(indices, reverse_track_ids)
    else:
        extra = audio_features[indices]

    return np.concatenate([base, extra], axis=1)


def build_playlist_tensor(playlist_ids, track_map, embeddings, audio_features, reverse_track_ids=None, max_len=20):
    playlist_indices = [track_map[track_id] for track_id in playlist_ids if track_id in track_map]

    if not playlist_indices:
        raise ValueError("None of the provided playlist tracks were found in the stage1 track map.")

    playlist_indices = playlist_indices[-max_len:]
    features = _combine_track_features(
        playlist_indices,
        embeddings,
        audio_features,
        reverse_track_ids=reverse_track_ids,
    )

    if len(features) < max_len:
        pad = np.zeros((max_len - len(features), features.shape[1]), dtype=np.float32)
        features = np.vstack([pad, features])

    playlist_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return playlist_tensor, set(playlist_indices)


def _top_indices(scores, top_k):
    top_k = min(int(top_k), scores.shape[0])

    if top_k <= 0:
        return np.asarray([], dtype=np.int64)

    if top_k == scores.shape[0]:
        return np.argsort(-scores)

    unordered = np.argpartition(scores, -top_k)[-top_k:]
    return unordered[np.argsort(-scores[unordered])]


def build_stage1_track_vector_cache(
    assets=None,
    output_path=STAGE1_TRACK_VECTORS_NPY,
    chunk_size=4096,
    dtype=np.float16,
    progress_callback=None,
):
    if assets is None:
        assets = load_stage1_assets()

    model = assets["model"]
    device = assets["device"]
    embeddings = assets["embeddings"]
    audio_features = assets["audio_features"]
    reverse_track_ids = assets["reverse_track_ids"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")

    sample = _combine_track_features(
        np.asarray([0], dtype=np.int64),
        embeddings,
        audio_features,
        reverse_track_ids=reverse_track_ids,
    )

    with torch.no_grad():
        encoded_sample = model.encode_tracks(torch.tensor(sample, dtype=torch.float32, device=device))

    output_dim = int(encoded_sample.shape[1])
    track_vectors = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=(embeddings.shape[0], output_dim),
    )

    model.eval()

    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])
        indices = np.arange(start, stop, dtype=np.int64)
        chunk_features = _combine_track_features(
            indices,
            embeddings,
            audio_features,
            reverse_track_ids=reverse_track_ids,
        )
        chunk = torch.tensor(chunk_features, dtype=torch.float32, device=device)

        with torch.no_grad():
            encoded = model.encode_tracks(chunk).detach().cpu().numpy()

        track_vectors[start:stop] = encoded.astype(dtype, copy=False)

        if progress_callback is not None:
            progress_callback(stop, embeddings.shape[0])

    track_vectors.flush()
    del track_vectors
    tmp_path.replace(output_path)

    assets["track_vectors"] = np.load(output_path, mmap_mode="r")
    return output_path


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
    audio_features = assets["audio_features"]
    track_map = assets["track_map"]
    reverse_track_map = assets["reverse_track_map"]
    reverse_track_ids = assets["reverse_track_ids"]
    track_vectors = assets.get("track_vectors")

    playlist_tensor, seen_indices = build_playlist_tensor(
        playlist_ids=playlist_ids,
        track_map=track_map,
        embeddings=embeddings,
        audio_features=audio_features,
        reverse_track_ids=reverse_track_ids,
        max_len=max_len,
    )
    playlist_tensor = playlist_tensor.to(device)

    with torch.no_grad():
        playlist_vec = model.encode_playlist(playlist_tensor)

    if track_vectors is not None:
        playlist_vec_np = playlist_vec.squeeze(0).detach().cpu().numpy().astype(np.float32)
        scores = np.empty(track_vectors.shape[0], dtype=np.float32)
        vector_chunk_size = max(chunk_size, 65_536)

        for start in range(0, track_vectors.shape[0], vector_chunk_size):
            stop = min(start + vector_chunk_size, track_vectors.shape[0])
            scores[start:stop] = np.asarray(track_vectors[start:stop], dtype=np.float32) @ playlist_vec_np

        for idx in seen_indices:
            scores[idx] = -np.inf

        top_indices = _top_indices(scores, top_k)

        return [
            {
                "rank": rank,
                "track_id": reverse_track_map[int(idx)],
                "mapped_idx": int(idx),
                "score": float(scores[idx]),
            }
            for rank, idx in enumerate(top_indices, start=1)
        ]

    scores = []

    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])
        chunk = torch.tensor(
            _combine_track_features(
                np.arange(start, stop),
                embeddings,
                audio_features,
                reverse_track_ids=reverse_track_ids,
            ),
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            chunk_vec = model.encode_tracks(chunk)
            chunk_scores = torch.matmul(playlist_vec, chunk_vec.T).squeeze(0).cpu().numpy()

        scores.append(chunk_scores)

    scores = np.concatenate(scores)

    for idx in seen_indices:
        scores[idx] = -np.inf

    top_indices = _top_indices(scores, top_k)

    return [
        {
            "rank": rank,
            "track_id": reverse_track_map[int(idx)],
            "mapped_idx": int(idx),
            "score": float(scores[idx]),
        }
        for rank, idx in enumerate(top_indices, start=1)
    ]

