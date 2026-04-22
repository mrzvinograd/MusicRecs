import sys
from pathlib import Path

import duckdb


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    AUDIO_FEATURES_FILTERED_PARQUET,
    TOP_TRACKS_PARQUET,
    TRACK_AUDIO_FEATURES_PARQUET,
)


con = duckdb.connect()
con.execute("PRAGMA memory_limit='16GB'")

print("Filtering audio features...")

con.execute(
    """
    COPY (
        SELECT af.*
        FROM read_parquet('{track_audio}') af
        JOIN read_parquet('{top_tracks}') tt
        ON af.track_id = CAST(tt.track_rowid AS VARCHAR)
    )
    TO '{audio_filtered}'
    (FORMAT PARQUET)
    """.format(
        track_audio=TRACK_AUDIO_FEATURES_PARQUET.as_posix(),
        top_tracks=TOP_TRACKS_PARQUET.as_posix(),
        audio_filtered=AUDIO_FEATURES_FILTERED_PARQUET.as_posix(),
    )
)

print(f"Done. Saved to {AUDIO_FEATURES_FILTERED_PARQUET}")
