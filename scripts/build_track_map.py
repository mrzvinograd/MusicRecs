import sys
from pathlib import Path
import pickle

import duckdb


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PLAYLIST_TRACKS_PARQUET, TRACK_ID_MAP_PKL


TOP_K = 100_000

con = duckdb.connect()
con.execute("PRAGMA memory_limit='16GB'")

print("Counting track frequency...")

query = f"""
SELECT track_rowid, COUNT(*) as freq
FROM read_parquet('{PLAYLIST_TRACKS_PARQUET.as_posix()}')
WHERE track_rowid IS NOT NULL
GROUP BY track_rowid
ORDER BY freq DESC
LIMIT {TOP_K}
"""

rows = con.execute(query).fetchall()

print("Building map...")

track_map = {track: idx for idx, (track, _) in enumerate(rows)}

with open(TRACK_ID_MAP_PKL, "wb") as f:
    pickle.dump(track_map, f)

print("Done. Size:", len(track_map))
