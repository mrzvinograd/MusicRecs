import pickle

import torch

from config import FILTERED_TRACK_ID_MAP_PKL, TRANSFORMER_MODEL_PT
from stage2.models.transformer_model import TransformerModel


def load_stage2_assets(
    checkpoint_path=TRANSFORMER_MODEL_PT,
    track_map_path=FILTERED_TRACK_ID_MAP_PKL,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(track_map_path, "rb") as f:
        track_map = pickle.load(f)

    pad_idx = len(track_map)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_kwargs = checkpoint.get(
            "model_kwargs",
            {"vocab_size": checkpoint.get("vocab_size", len(track_map) + 1), "embed_dim": checkpoint.get("embed_dim", 256)},
        )
        pad_idx = checkpoint.get("pad_idx", pad_idx)
    else:
        state_dict = checkpoint
        model_kwargs = {"vocab_size": len(track_map) + 1, "embed_dim": 256}

    model = TransformerModel(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "device": device,
        "model": model,
        "track_map": track_map,
        "pad_idx": pad_idx,
    }
