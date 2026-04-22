import pickle

import duckdb
import numpy as np
import torch

from config import (
    PLAYLIST_TRACKS_FILTERED_PARQUET,
    TRACK_EMBEDDINGS_NPY,
    TRACK2VEC_TRACK_MAP_PKL,
)


class PlaylistDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        playlist_parquet=PLAYLIST_TRACKS_FILTERED_PARQUET,
        embeddings_file=TRACK_EMBEDDINGS_NPY,
        track_map_file=TRACK2VEC_TRACK_MAP_PKL,
        max_len=20,
        return_target_idx=False,
    ):
        print("Connecting to DuckDB...")

        self.con = duckdb.connect(database=":memory:")
        self.con.execute("PRAGMA memory_limit='8GB'")
        self.con.execute("PRAGMA threads=4")

        self.playlist_parquet = str(playlist_parquet)
        self.max_len = max_len
        self.return_target_idx = return_target_idx

        print("Loading embeddings...")
        self.embeddings = np.load(str(embeddings_file))

        print("Loading track map...")
        with open(track_map_file, "rb") as f:
            self.track_map = pickle.load(f)

        self.reverse_track_map = {idx: track for track, idx in self.track_map.items()}

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
            rows = cursor.fetchmany(10000)

            if not rows:
                break

            for pid, track in rows:
                idx = self.track_map.get(track)

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

        if len(current_playlist) > 1:
            sample = self.process_playlist(current_playlist)

            if sample is not None:
                yield sample

    def process_playlist(self, playlist_idx_list):
        if len(playlist_idx_list) < 2:
            return None

        input_tracks = playlist_idx_list[:-1][-self.max_len:]
        target_idx = playlist_idx_list[-1]

        features = self.embeddings[input_tracks]

        if len(features) < self.max_len:
            pad = np.zeros((self.max_len - len(features), features.shape[1]), dtype=np.float32)
            features = np.vstack([pad, features])

        playlist_features = torch.tensor(features, dtype=torch.float32)
        target_vec = torch.tensor(self.embeddings[target_idx], dtype=torch.float32)

        if self.return_target_idx:
            return playlist_features, target_vec, torch.tensor(target_idx, dtype=torch.long)

        return playlist_features, target_vec
