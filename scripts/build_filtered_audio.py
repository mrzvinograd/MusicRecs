import sys
from pathlib import Path

import duckdb


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    AUDIO_FEATURES_FILTERED_PARQUET,
    TRACKS_PARQUET,
    TRACK_AUDIO_FEATURES_PARQUET,
)


con = duckdb.connect()
con.execute("PRAGMA memory_limit='16GB'")


def detect_spotify_id_column():
    candidates = ("id", "spotify_id", "spotify_track_id", "uri")
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{TRACKS_PARQUET.as_posix()}')"
    ).fetchall()
    columns = {row[0] for row in rows}

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise RuntimeError(
        "Spotify track id column was not found in tracks.parquet. "
        "Expected one of: id, spotify_id, spotify_track_id, uri."
    )


print("Filtering audio features...")
spotify_id_column = detect_spotify_id_column()

con.execute(
    """
    COPY (
        SELECT
            tr.rowid AS track_rowid,
            af.*
        FROM read_parquet('{track_audio}') af
        JOIN read_parquet('{tracks}') tr
        ON af.track_id = CAST(tr.{spotify_id_column} AS VARCHAR)
    )
    TO '{audio_filtered}'
    (FORMAT PARQUET)
    """.format(
        track_audio=TRACK_AUDIO_FEATURES_PARQUET.as_posix(),
        tracks=TRACKS_PARQUET.as_posix(),
        spotify_id_column=spotify_id_column,
        audio_filtered=AUDIO_FEATURES_FILTERED_PARQUET.as_posix(),
    )
)

print(f"Done. Saved to {AUDIO_FEATURES_FILTERED_PARQUET}")
