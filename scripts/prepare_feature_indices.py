import sys
from pathlib import Path

import duckdb
import numpy as np
from numpy.lib.format import open_memmap


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import AUDIO_FEATURES_FILTERED_PARQUET, AUDIO_FEATURES_NPY, AUDIO_ROWIDS_NPY


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

BATCH_SIZE = 20_000


def cast_expr(column):
    return f"COALESCE(TRY_CAST({column} AS DOUBLE), 0.0)"


con = duckdb.connect()
con.execute("PRAGMA memory_limit='16GB'")
con.execute("PRAGMA threads=4")
con.execute("PRAGMA preserve_insertion_order=false")

print("Loading filtered audio with DuckDB...")

stats_query = ",\n".join(
    [
        f"AVG({cast_expr(column)}) AS {column}_mean,\n"
        f"STDDEV_POP({cast_expr(column)}) AS {column}_std"
        for column in FEATURE_COLUMNS
    ]
)

print("Pass 1/2: computing normalization statistics...")
stats_row = con.execute(
    f"""
    SELECT
        {stats_query}
    FROM read_parquet('{AUDIO_FEATURES_FILTERED_PARQUET.as_posix()}')
    WHERE track_rowid IS NOT NULL
    """
).fetchone()

normalization = {}

for idx, column in enumerate(FEATURE_COLUMNS):
    mean_value = float(stats_row[idx * 2] or 0.0)
    std_value = float(stats_row[idx * 2 + 1] or 1.0)

    if std_value == 0.0:
        std_value = 1.0

    normalization[column] = (mean_value, std_value)

normalized_columns = ",\n".join(
    [
        (
            f"CAST((({cast_expr(column)} - {normalization[column][0]}) / {normalization[column][1]}) AS FLOAT) "
            f"AS {column}"
        )
        for column in FEATURE_COLUMNS
    ]
)

print("Pass 2/2: streaming normalized audio features...")
row_count = con.execute(
    f"""
    SELECT COUNT(*)
    FROM read_parquet('{AUDIO_FEATURES_FILTERED_PARQUET.as_posix()}')
    WHERE track_rowid IS NOT NULL
    """
).fetchone()[0]

cursor = con.execute(
    f"""
    SELECT
        CAST(track_rowid AS BIGINT) AS track_rowid,
        {normalized_columns}
    FROM read_parquet('{AUDIO_FEATURES_FILTERED_PARQUET.as_posix()}')
    WHERE track_rowid IS NOT NULL
    ORDER BY track_rowid
    """
)

rowid_memmap = open_memmap(AUDIO_ROWIDS_NPY, mode="w+", dtype=np.int64, shape=(row_count,))
feature_memmap = open_memmap(
    AUDIO_FEATURES_NPY,
    mode="w+",
    dtype=np.float32,
    shape=(row_count, len(FEATURE_COLUMNS)),
)
offset = 0

while True:
    batch = cursor.fetchmany(BATCH_SIZE)

    if not batch:
        break

    rowids = np.fromiter((row[0] for row in batch), dtype=np.int64, count=len(batch))
    features = np.asarray([row[1:] for row in batch], dtype=np.float32)
    next_offset = offset + rowids.shape[0]
    rowid_memmap[offset:next_offset] = rowids
    feature_memmap[offset:next_offset] = features
    offset = next_offset

print("Saving compact audio feature arrays...")
rowid_memmap.flush()
feature_memmap.flush()

con.close()

print(f"Done. Saved to {AUDIO_ROWIDS_NPY} and {AUDIO_FEATURES_NPY}")
