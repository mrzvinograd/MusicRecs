import sys
from pathlib import Path

import duckdb
from gensim.models import Word2Vec

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import PLAYLIST_TRACKS_FILTERED_PARQUET, TRACK2VEC_MODEL


PARQUET = PLAYLIST_TRACKS_FILTERED_PARQUET

BATCH = 50000


class PlaylistIterator:

    def __init__(self):

        self.conn = duckdb.connect()
        self.conn.execute("PRAGMA memory_limit='16GB'")

        self.cursor = self.conn.execute(f"""
            SELECT playlist_rowid, track_rowid
            FROM read_parquet('{PARQUET.as_posix()}')
            ORDER BY playlist_rowid, position
        """)

    def __iter__(self):

        last = None
        session = []

        while True:

            rows = self.cursor.fetchmany(BATCH)

            if not rows:
                break

            for pid, track in rows:

                if last and pid != last:

                    if len(session) > 1:
                        yield session

                    session = []

                session.append(str(track))
                last = pid

        if len(session) > 1:
            yield session


sentences = PlaylistIterator()

model = Word2Vec(
    vector_size=512,
    window=5,
    min_count=3,
    workers=12,
    sg=1,
    negative=10,
)

model.build_vocab(sentences)

model.train(
    sentences,
    total_examples=model.corpus_count,
    epochs=10
)

model.save(str(TRACK2VEC_MODEL))
