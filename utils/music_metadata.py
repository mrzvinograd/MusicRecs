import duckdb

from config import ALBUMS_PARQUET, ARTISTS_PARQUET, TRACKS_PARQUET, TRACK_ARTISTS_PARQUET, TRACK_SEARCH_INDEX_PARQUET


def _connect():
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=2")
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
        if spotify_id_column:
            spotify_id_sql = f"CAST(tr.{spotify_id_column} AS VARCHAR) AS spotify_track_id,"
        else:
            spotify_id_sql = "NULL AS spotify_track_id,"

        con.execute(
            f"""
            COPY (
                WITH artist_names AS (
                    SELECT
                        ta.track_rowid,
                        string_agg(ar.name, ', ' ORDER BY ar.name) AS artist_names
                    FROM read_parquet('{TRACK_ARTISTS_PARQUET.as_posix()}') ta
                    JOIN read_parquet('{ARTISTS_PARQUET.as_posix()}') ar
                        ON ta.artist_rowid = ar.rowid
                    GROUP BY ta.track_rowid
                )
                SELECT
                    tr.rowid AS track_rowid,
                    {spotify_id_sql}
                    tr.name AS track_name,
                    COALESCE(an.artist_names, 'Unknown Artist') AS artist_names,
                    COALESCE(al.name, 'Unknown Album') AS album_name
                FROM read_parquet('{TRACKS_PARQUET.as_posix()}') tr
                LEFT JOIN artist_names an
                    ON tr.rowid = an.track_rowid
                LEFT JOIN read_parquet('{ALBUMS_PARQUET.as_posix()}') al
                    ON tr.album_rowid = al.rowid
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

    require_track_search_index()
    ids_sql = ", ".join(str(int(track_id)) for track_id in track_ids)
    con = _connect()

    try:
        query = f"""
        SELECT track_rowid, track_name, artist_names, album_name
        FROM read_parquet('{TRACK_SEARCH_INDEX_PARQUET.as_posix()}') meta
        WHERE track_rowid IN ({ids_sql})
        """
        rows = con.execute(query).fetchall()
    finally:
        con.close()

    return {
        int(track_rowid): {
            "track_name": track_name,
            "artist_names": artist_names,
            "album_name": album_name,
        }
        for track_rowid, track_name, artist_names, album_name in rows
    }


def search_tracks(query_text, limit=20):
    query_text = (query_text or "").strip()

    if len(query_text) < 2:
        return []

    require_track_search_index()
    safe_query = query_text.replace("'", "''")
    con = _connect()

    try:
        query = f"""
        SELECT track_rowid, track_name, artist_names, album_name
        FROM read_parquet('{TRACK_SEARCH_INDEX_PARQUET.as_posix()}') meta
        WHERE lower(track_name) LIKE lower('%{safe_query}%')
           OR lower(artist_names) LIKE lower('%{safe_query}%')
           OR lower(album_name) LIKE lower('%{safe_query}%')
        ORDER BY track_name, artist_names
        LIMIT {int(limit)}
        """
        rows = con.execute(query).fetchall()
    finally:
        con.close()

    return [
        {
            "track_id": int(track_rowid),
            "track_name": track_name,
            "artist_names": artist_names,
            "album_name": album_name,
        }
        for track_rowid, track_name, artist_names, album_name in rows
    ]


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
            f"Track token '{token}' was not found. Use numeric track_rowid or a known Spotify track id from the dataset."
        )

    return int(row[0])
