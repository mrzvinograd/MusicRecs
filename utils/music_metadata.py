import duckdb

from config import ALBUMS_PARQUET, ARTISTS_PARQUET, TRACKS_PARQUET, TRACK_ARTISTS_PARQUET, TRACK_SEARCH_INDEX_PARQUET


def _connect():
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='16GB'")
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA preserve_insertion_order=false")
    return con


def _parquet_columns(con, parquet_path):
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path.as_posix()}')").fetchall()
    return {row[0] for row in rows}


def _detect_spotify_id_column(con):
    candidates = ("id", "spotify_id", "spotify_track_id", "uri")
    columns = _parquet_columns(con, TRACKS_PARQUET)

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def ensure_track_search_index():
    if TRACK_SEARCH_INDEX_PARQUET.exists():
        return TRACK_SEARCH_INDEX_PARQUET

    con = _connect()

    try:
        spotify_id_column = _detect_spotify_id_column(con)
        if not spotify_id_column:
            raise RuntimeError(
                "Spotify track id column was not found in tracks.parquet. "
                "Expected one of: id, spotify_id, spotify_track_id, uri."
            )

        spotify_id_expr = f"CAST(tr.{spotify_id_column} AS VARCHAR)"
        normalized_spotify_id_expr = (
            f"regexp_extract({spotify_id_expr}, '([A-Za-z0-9]{{22}})$', 1)"
        )

        con.execute(
            f"""
            COPY (
                SELECT
                    tr.rowid AS track_rowid,
                    CASE
                        WHEN length({normalized_spotify_id_expr}) = 22 THEN {normalized_spotify_id_expr}
                        ELSE {spotify_id_expr}
                    END AS spotify_track_id
                FROM read_parquet('{TRACKS_PARQUET.as_posix()}') tr
                WHERE {spotify_id_expr} IS NOT NULL
            )
            TO '{TRACK_SEARCH_INDEX_PARQUET.as_posix()}'
            (FORMAT PARQUET)
            """
        )
    finally:
        con.close()

    return TRACK_SEARCH_INDEX_PARQUET


def require_track_search_index():
    if not TRACK_SEARCH_INDEX_PARQUET.exists():
        raise FileNotFoundError(
            f"Track search index not found: {TRACK_SEARCH_INDEX_PARQUET}. "
            "Build it first with `python scripts/build_track_search_index.py`."
        )

    return TRACK_SEARCH_INDEX_PARQUET


def fetch_track_metadata(track_ids):
    if not track_ids:
        return {}

    ids_sql = ", ".join(str(int(track_id)) for track_id in track_ids)
    con = _connect()

    try:
        query = f"""
        WITH selected_tracks AS (
            SELECT rowid, name, album_rowid
            FROM read_parquet('{TRACKS_PARQUET.as_posix()}')
            WHERE rowid IN ({ids_sql})
        ),
        artist_names AS (
            SELECT
                ta.track_rowid,
                string_agg(ar.name, ', ' ORDER BY ar.name) AS artist_names
            FROM read_parquet('{TRACK_ARTISTS_PARQUET.as_posix()}') ta
            JOIN read_parquet('{ARTISTS_PARQUET.as_posix()}') ar
                ON ta.artist_rowid = ar.rowid
            WHERE ta.track_rowid IN ({ids_sql})
            GROUP BY ta.track_rowid
        )
        SELECT
            st.rowid AS track_rowid,
            st.name AS track_name,
            COALESCE(an.artist_names, 'Unknown Artist') AS artist_names,
            al.name AS album_name,
            idx.spotify_track_id
        FROM selected_tracks st
        LEFT JOIN artist_names an
            ON st.rowid = an.track_rowid
        LEFT JOIN read_parquet('{ALBUMS_PARQUET.as_posix()}') al
            ON st.album_rowid = al.rowid
        LEFT JOIN read_parquet('{TRACK_SEARCH_INDEX_PARQUET.as_posix()}') idx
            ON st.rowid = idx.track_rowid
        """
        rows = con.execute(query).fetchall()
    finally:
        con.close()

    return {
        int(track_rowid): {
            "track_name": track_name,
            "artist_names": artist_names,
            "album_name": album_name,
            "spotify_track_id": spotify_track_id,
        }
        for track_rowid, track_name, artist_names, album_name, spotify_track_id in rows
    }


def resolve_track_token(token):
    token = (token or "").strip()

    if not token:
        return None

    if token.isdigit():
        return int(token)

    require_track_search_index()
    safe_token = token.replace("'", "''")
    con = _connect()

    try:
        row = con.execute(
            f"""
            SELECT track_rowid
            FROM read_parquet('{TRACK_SEARCH_INDEX_PARQUET.as_posix()}')
            WHERE spotify_track_id = '{safe_token}'
            LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()

    if row is None:
        raise ValueError(
            f"Spotify track id '{token}' was not found in the dataset index."
        )

    return int(row[0])
