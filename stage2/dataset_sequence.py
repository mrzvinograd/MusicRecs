import torch
from torch.utils.data import IterableDataset
import duckdb


class SequenceDataset(IterableDataset):

    def __init__(self, seq_len, db_path, track_map):

        self.seq_len = seq_len
        self.db_path = db_path
        self.track_map = track_map

    def __iter__(self):

        con = duckdb.connect()
        con.execute("PRAGMA memory_limit='16GB'")

        print("Streaming playlists...")

        query = f"""
        SELECT playlist_rowid, track_rowid
        FROM read_parquet('{self.db_path}')
        WHERE track_rowid IS NOT NULL
        ORDER BY playlist_rowid, position
        """

        cursor = con.execute(query)

        current_playlist = []
        last_pid = None

        while True:

            batch = cursor.fetchmany(100000)  # 🔥 НЕ грузим всё в память

            if not batch:
                break

            for pid, track in batch:

                # фильтр по vocab
                if track not in self.track_map:
                    continue

                track_id = self.track_map[track]

                # новый плейлист
                if last_pid is not None and pid != last_pid:

                    yield from self.process_playlist(current_playlist)
                    current_playlist = []

                current_playlist.append(track_id)
                last_pid = pid

        # последний плейлист
        if len(current_playlist) > self.seq_len:
            yield from self.process_playlist(current_playlist)

    def process_playlist(self, playlist):

        if len(playlist) <= self.seq_len:
            return

        for i in range(len(playlist) - self.seq_len):

            seq = playlist[i:i + self.seq_len]
            target = playlist[i + self.seq_len]

            yield (
                torch.tensor(seq, dtype=torch.long),
                torch.tensor(target, dtype=torch.long)
            )
