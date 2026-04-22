import sys
from pathlib import Path

import duckdb


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PLAYLIST_TRACKS_PARQUET, TOP_TRACKS_PARQUET


con = duckdb.connect()
con.execute("PRAGMA memory_limit='16GB'")

print("Counting track frequency...")

con.execute(
    f"""
    CREATE TABLE track_freq AS
    SELECT
        track_rowid,
        COUNT(*) as freq
    FROM read_parquet('{PLAYLIST_TRACKS_PARQUET.as_posix()}')
    WHERE track_rowid IS NOT NULL
    GROUP BY track_rowid
    """
)

print("Selecting top tracks...")

con.execute(
    """
    CREATE TABLE top_tracks AS
    SELECT *
    FROM track_freq
    ORDER BY freq DESC
    LIMIT 1000000
    """
)

con.execute(f"COPY top_tracks TO '{TOP_TRACKS_PARQUET.as_posix()}' (FORMAT PARQUET)")

print(f"Done. Saved to {TOP_TRACKS_PARQUET}")
