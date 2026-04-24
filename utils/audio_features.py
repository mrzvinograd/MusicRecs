import pickle

import numpy as np

from config import AUDIO_INDEX_PKL


def load_aligned_audio_features(track_map, audio_index_path=AUDIO_INDEX_PKL):
    if not audio_index_path.exists():
        return None

    with open(audio_index_path, "rb") as f:
        audio_index = pickle.load(f)

    if not audio_index:
        return None

    sample = next(iter(audio_index.values()))
    feature_dim = len(sample)
    aligned = np.zeros((len(track_map), feature_dim), dtype=np.float32)

    for raw_track_id, mapped_idx in track_map.items():
        values = audio_index.get(str(raw_track_id))

        if values is None:
            values = audio_index.get(raw_track_id)

        if values is None:
            continue

        aligned[mapped_idx] = np.asarray(values, dtype=np.float32)

    return aligned
