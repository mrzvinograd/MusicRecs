import sys
from pathlib import Path
import pickle

import polars as pl


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


print("Loading filtered audio...")

audio = pl.read_parquet(AUDIO_FEATURES_FILTERED_PARQUET)
audio = audio.select(["track_id", *FEATURE_COLUMNS]).drop_nulls(subset=["track_id"])

normalized_columns = []

for column in FEATURE_COLUMNS:
    col_expr = pl.col(column).cast(pl.Float32).fill_null(0.0)
    stats = audio.select(
        pl.mean(column).alias("mean"),
        pl.std(column).alias("std"),
    ).row(0)
    mean_value = float(stats[0]) if stats[0] is not None else 0.0
    std_value = float(stats[1]) if stats[1] not in (None, 0.0) else 1.0
    normalized_columns.append(((col_expr - mean_value) / std_value).alias(column))

audio = audio.select(
    pl.col("track_id"),
    *normalized_columns,
)

print("Building index...")

audio_index = {}

for row in audio.iter_rows(named=True):
    audio_index[row["track_id"]] = [float(row[column]) for column in FEATURE_COLUMNS]

print("Saving...")

with open(AUDIO_INDEX_PKL, "wb") as f:
    pickle.dump(audio_index, f)

print(f"Done. Saved to {AUDIO_INDEX_PKL}")
