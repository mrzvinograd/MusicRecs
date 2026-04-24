import pickle

import numpy as np

from config import AUDIO_FEATURES_NPY, AUDIO_INDEX_PKL, AUDIO_ROWIDS_NPY


def _load_from_npy(track_map):
    if not AUDIO_ROWIDS_NPY.exists() or not AUDIO_FEATURES_NPY.exists():
        return None

    rowids = np.load(AUDIO_ROWIDS_NPY, mmap_mode="r")
    features = np.load(AUDIO_FEATURES_NPY, mmap_mode="r")

    if rowids.size == 0 or features.size == 0:
        return None

    aligned = np.zeros((len(track_map), features.shape[1]), dtype=np.float32)
    sorted_pos = np.searchsorted(rowids, np.fromiter(track_map.keys(), dtype=np.int64))
    keys = list(track_map.keys())

    for raw_track_id, pos in zip(keys, sorted_pos):
        if pos >= rowids.shape[0] or int(rowids[pos]) != int(raw_track_id):
            continue

        aligned[track_map[raw_track_id]] = np.asarray(features[pos], dtype=np.float32)

    return aligned


def _load_from_pickle(track_map, audio_index_path=AUDIO_INDEX_PKL):
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


def load_aligned_audio_features(track_map, audio_index_path=AUDIO_INDEX_PKL):
    aligned = _load_from_npy(track_map)

    if aligned is not None:
        return aligned

    return _load_from_pickle(track_map, audio_index_path=audio_index_path)
