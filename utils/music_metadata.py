import duckdb

from config import ALBUMS_PARQUET, ARTISTS_PARQUET, TRACKS_PARQUET, TRACK_ARTISTS_PARQUET


def fetch_track_metadata(track_ids):
    if not track_ids:
        return {}

    ids_sql = ", ".join(str(int(track_id)) for track_id in track_ids)
    con = duckdb.connect()

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
        al.name AS album_name
    FROM selected_tracks st
    LEFT JOIN artist_names an
        ON st.rowid = an.track_rowid
    LEFT JOIN read_parquet('{ALBUMS_PARQUET.as_posix()}') al
        ON st.album_rowid = al.rowid
    """

    rows = con.execute(query).fetchall()

    return {
        int(track_rowid): {
            "track_name": track_name,
            "artist_names": artist_names,
            "album_name": album_name,
        }
        for track_rowid, track_name, artist_names, album_name in rows
    }
