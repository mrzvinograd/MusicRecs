import sys
from pathlib import Path

import duckdb
from gensim.models import Word2Vec


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PLAYLIST_TRACKS_FILTERED_PARQUET, TRACK2VEC_MODEL


model = Word2Vec.load(str(TRACK2VEC_MODEL))
con = duckdb.connect()

playlist = con.execute(
    f"""
    SELECT track_rowid
    FROM read_parquet('{PLAYLIST_TRACKS_FILTERED_PARQUET.as_posix()}')
    LIMIT 20
    """
).fetchall()

tracks = [str(x[0]) for x in playlist]

print("Playlist tracks:")
print(tracks)

print("\nPredicted similar tracks:")

for track_id in tracks[:5]:
    print("\nTrack:", track_id)

    try:
        sim = model.wv.most_similar(track_id, topn=5)

        for item in sim:
            print(item)

    except KeyError:
        print("Track not in vocab")
