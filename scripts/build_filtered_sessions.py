import sys
from pathlib import Path

import duckdb


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    PLAYLIST_TRACKS_FILTERED_PARQUET,
    PLAYLIST_TRACKS_PARQUET,
    TOP_TRACKS_PARQUET,
)


con = duckdb.connect()
con.execute("PRAGMA memory_limit='16GB'")

print("Filtering playlists...")

con.execute(
    """
    COPY (
        SELECT pt.playlist_rowid,
               pt.position,
               pt.track_rowid
        FROM read_parquet('{playlist_tracks}') pt
        JOIN read_parquet('{top_tracks}') tt
        ON pt.track_rowid = tt.track_rowid
        ORDER BY playlist_rowid, position
    )
    TO '{playlist_filtered}'
    (FORMAT PARQUET)
    """.format(
        playlist_tracks=PLAYLIST_TRACKS_PARQUET.as_posix(),
        top_tracks=TOP_TRACKS_PARQUET.as_posix(),
        playlist_filtered=PLAYLIST_TRACKS_FILTERED_PARQUET.as_posix(),
    )
)

print(f"Done. Saved to {PLAYLIST_TRACKS_FILTERED_PARQUET}")
