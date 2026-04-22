import sys
from pathlib import Path
import pickle

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    TRACK2VEC_VECTORS,
    TRACK_EMBEDDINGS_NPY,
    TRACK_IDS_TXT,
    TRACK2VEC_TRACK_MAP_PKL,
)


print("Loading exported embeddings...")
embeddings = np.load(TRACK_EMBEDDINGS_NPY, mmap_mode="r")

print("Loading raw Word2Vec vectors...")
wv_vectors = np.load(TRACK2VEC_VECTORS, mmap_mode="r")

print("Loading track ids...")
with open(TRACK_IDS_TXT, "r", encoding="utf-8") as f:
    track_ids = [int(line.strip()) for line in f if line.strip()]

print("Loading track map...")
with open(TRACK2VEC_TRACK_MAP_PKL, "rb") as f:
    track_map = pickle.load(f)

embedding_rows = int(embeddings.shape[0])
wv_rows = int(wv_vectors.shape[0])
ids_count = len(track_ids)
map_size = len(track_map)

print(f"Exported embedding rows: {embedding_rows}")
print(f"Word2Vec vector rows: {wv_rows}")
print(f"Track ids count: {ids_count}")
print(f"Track map size: {map_size}")

if not (embedding_rows == wv_rows == ids_count == map_size):
    raise RuntimeError(
        "Stage0 artifact size mismatch between exported embeddings, raw Word2Vec vectors, ids, and track map."
    )

if embeddings.shape[1] != wv_vectors.shape[1]:
    raise RuntimeError(
        f"Embedding dimension mismatch: exported={embeddings.shape[1]}, word2vec={wv_vectors.shape[1]}"
    )

sample_size = min(1000, embedding_rows)

for idx, raw_track_id in enumerate(track_ids[:sample_size]):
    mapped_idx = track_map.get(raw_track_id)

    if mapped_idx != idx:
        raise RuntimeError(
            f"Track map mismatch at position {idx}: track_id={raw_track_id}, mapped_idx={mapped_idx}"
        )

    if not np.allclose(embeddings[idx], wv_vectors[idx]):
        raise RuntimeError(
            f"Vector mismatch at position {idx}: track_id={raw_track_id}"
        )

print("Stage0 artifacts are consistent.")
