import duckdb
import torch
import numpy as np
import pickle

from config import (
    PLAYLIST_TRACKS_FILTERED_PARQUET,
    TRACK_EMBEDDINGS_NPY,
    TRACK_ID_MAP_PKL,
)


class PlaylistDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 playlist_parquet=PLAYLIST_TRACKS_FILTERED_PARQUET,
                 embeddings_file=TRACK_EMBEDDINGS_NPY,
                 track_map_file=TRACK_ID_MAP_PKL):

        print("Connecting to DuckDB...")

        self.con = duckdb.connect(database=':memory:')
        self.con.execute("PRAGMA memory_limit='8GB'")
        self.con.execute("PRAGMA threads=4")

        self.playlist_parquet = str(playlist_parquet)

        print("Loading embeddings...")
        self.embeddings = np.load(str(embeddings_file))

        print("Loading track map...")
        with open(track_map_file, "rb") as f:
            self.track_map = pickle.load(f)

        self.MAX_LEN = 20  # длина последовательности

    def __iter__(self):

        print("Streaming playlists...")

        query = f"""
            SELECT playlist_rowid, track_rowid
            FROM read_parquet('{self.playlist_parquet}')
            ORDER BY playlist_rowid, position
        """

        cursor = self.con.execute(query)

        current_playlist = []
        last_pid = None

        while True:

            rows = cursor.fetchmany(10000)  # 🔥 ключевая оптимизация

            if not rows:
                break

            for pid, track in rows:

                idx = self.track_map.get(track)

                # пропускаем треки без embedding
                if idx is None:
                    continue

                if last_pid is not None and pid != last_pid:

                    if len(current_playlist) > 1:

                        sample = self.process_playlist(current_playlist)

                        if sample is not None:
                            yield sample

                    current_playlist = []

                current_playlist.append(idx)
                last_pid = pid

        # последний плейлист
        if len(current_playlist) > 1:

            sample = self.process_playlist(current_playlist)

            if sample is not None:
                yield sample

    def process_playlist(self, playlist_idx_list):

        if len(playlist_idx_list) < 2:
            return None

        input_tracks = playlist_idx_list[:-1][-self.MAX_LEN:]
        target_idx = playlist_idx_list[-1]

        features = self.embeddings[input_tracks]

        # 🔥 padding
        if len(features) < self.MAX_LEN:

            pad = np.zeros((self.MAX_LEN - len(features), features.shape[1]))
            features = np.vstack([pad, features])

        features = torch.tensor(features, dtype=torch.float32)
        target_vec = torch.tensor(
            self.embeddings[target_idx],
            dtype=torch.float32
        )

        return features, target_vec
