import math
import pickle
import sys
from pathlib import Path

import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import AUDIO_FEATURES_FILTERED_PARQUET, AUDIO_INDEX_PKL


FEATURE_COLUMNS = [
    "duration_ms",
    "time_signature",
    "tempo",
    "key",
    "mode",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

BATCH_SIZE = 50_000


def _safe_float(value):
    if value is None:
        return 0.0

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if math.isnan(value) or math.isinf(value):
        return 0.0

    return value


print("Scanning filtered audio parquet...")
parquet = pq.ParquetFile(AUDIO_FEATURES_FILTERED_PARQUET)

print("Pass 1/2: computing normalization statistics...")
feature_stats = {
    column: {"count": 0, "sum": 0.0, "sum_sq": 0.0}
    for column in FEATURE_COLUMNS
}

for batch in parquet.iter_batches(columns=["track_rowid", *FEATURE_COLUMNS], batch_size=BATCH_SIZE):
    rows = batch.to_pylist()

    for row in rows:
        if row["track_rowid"] is None:
            continue

        for column in FEATURE_COLUMNS:
            value = _safe_float(row.get(column))
            stats = feature_stats[column]
            stats["count"] += 1
            stats["sum"] += value
            stats["sum_sq"] += value * value

normalization = {}

for column, stats in feature_stats.items():
    count = max(stats["count"], 1)
    mean_value = stats["sum"] / count
    variance = max((stats["sum_sq"] / count) - (mean_value * mean_value), 0.0)
    std_value = math.sqrt(variance) if variance > 0.0 else 1.0
    normalization[column] = (mean_value, std_value)

print("Pass 2/2: building normalized audio index...")
audio_index = {}

for batch in parquet.iter_batches(columns=["track_rowid", *FEATURE_COLUMNS], batch_size=BATCH_SIZE):
    rows = batch.to_pylist()

    for row in rows:
        track_rowid = row["track_rowid"]

        if track_rowid is None:
            continue

        normalized = []

        for column in FEATURE_COLUMNS:
            value = _safe_float(row.get(column))
            mean_value, std_value = normalization[column]
            normalized.append((value - mean_value) / std_value)

        audio_index[int(track_rowid)] = normalized

print("Saving...")

with open(AUDIO_INDEX_PKL, "wb") as f:
    pickle.dump(audio_index, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Done. Saved to {AUDIO_INDEX_PKL}")
