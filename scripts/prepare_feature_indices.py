import pickle
import sys
from pathlib import Path

import duckdb


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

BATCH_SIZE = 100_000


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
            f"(({cast_expr(column)} - {normalization[column][0]}) / {normalization[column][1]}) "
            f"AS {column}"
        )
        for column in FEATURE_COLUMNS
    ]
)

print("Pass 2/2: building normalized audio index...")
cursor = con.execute(
    f"""
    SELECT
        CAST(track_rowid AS BIGINT) AS track_rowid,
        {normalized_columns}
    FROM read_parquet('{AUDIO_FEATURES_FILTERED_PARQUET.as_posix()}')
    WHERE track_rowid IS NOT NULL
    """
)

audio_index = {}

while True:
    batch = cursor.fetchmany(BATCH_SIZE)

    if not batch:
        break

    for row in batch:
        track_rowid = int(row[0])
        audio_index[track_rowid] = [float(value) for value in row[1:]]

print("Saving...")

with open(AUDIO_INDEX_PKL, "wb") as f:
    pickle.dump(audio_index, f, protocol=pickle.HIGHEST_PROTOCOL)

con.close()

print(f"Done. Saved to {AUDIO_INDEX_PKL}")
