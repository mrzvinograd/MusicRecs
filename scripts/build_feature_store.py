import sys
from pathlib import Path

import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ALBUMS_PARQUET,
    ARTISTS_PARQUET,
    ARTIST_GENRES_PARQUET,
    TRACKS_PARQUET,
    TRACK_ARTISTS_PARQUET,
    TRACK_AUDIO_FEATURES_PARQUET,
    TRACK_FEATURE_STORE_PARQUET,
)


tracks = pl.scan_parquet(TRACKS_PARQUET)
audio = pl.scan_parquet(TRACK_AUDIO_FEATURES_PARQUET)
track_artists = pl.scan_parquet(TRACK_ARTISTS_PARQUET)
artists = pl.scan_parquet(ARTISTS_PARQUET)
genres = pl.scan_parquet(ARTIST_GENRES_PARQUET)
albums = pl.scan_parquet(ALBUMS_PARQUET)

df = (
    tracks
    .join(audio, left_on="rowid", right_on="track_id")
    .join(track_artists, left_on="rowid", right_on="track_rowid")
    .join(artists, left_on="artist_rowid", right_on="rowid")
    .join(albums, left_on="album_rowid", right_on="rowid")
    .join(genres, on="artist_rowid", how="left")
)

df.select([
    "rowid",
    "danceability",
    "energy",
    "tempo",
    "valence",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "followers_total",
    "duration_ms",
    "total_tracks",
    "genre",
]).collect().write_parquet(TRACK_FEATURE_STORE_PARQUET)

print(f"Feature store created at {TRACK_FEATURE_STORE_PARQUET}")
