import random

import duckdb
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from config import TRACK_EMBEDDINGS_NPY, TRACK_IDS_TXT


MAX_CONTEXT_LEN = 20


def ranking_collate(batch, pad_idx):
    playlists, tracks, labels = zip(*batch)

    playlists = pad_sequence(
        playlists,
        batch_first=True,
        padding_value=pad_idx,
    )

    return playlists, torch.stack(tracks), torch.stack(labels)


class HardNegativeMixin:

    def setup_negative_sampling(
        self,
        track_map,
        embedding_file=None,
        embedding_ids_file=None,
        use_hard_negatives=False,
        hard_negative_ratio=0.5,
        hard_pool_size=128,
    ):
        self.track_map = track_map
        self.all_tracks = list(track_map.values())
        self.embedding_file = embedding_file
        self.embedding_ids_file = embedding_ids_file
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_pool_size = hard_pool_size
        self.hard_negative_cache = {}
        self.track_embeddings = None

        if not self.use_hard_negatives:
            return

        if not self.embedding_file or not self.embedding_ids_file:
            self.use_hard_negatives = False
            return

        try:
            raw_embeddings = np.load(self.embedding_file, mmap_mode="r")
            with open(self.embedding_ids_file, "r", encoding="utf-8") as f:
                embedding_track_ids = [int(line.strip()) for line in f if line.strip()]
        except OSError:
            self.use_hard_negatives = False
            return

        embedding_index_by_track = {
            track_id: idx for idx, track_id in enumerate(embedding_track_ids)
        }

        emb_dim = raw_embeddings.shape[1]
        track_embeddings = np.zeros((len(track_map), emb_dim), dtype=np.float32)
        valid_tracks = []

        for model_track_id, mapped_idx in track_map.items():
            embedding_idx = embedding_index_by_track.get(model_track_id)

            if embedding_idx is None:
                continue

            vector = np.asarray(raw_embeddings[embedding_idx], dtype=np.float32)
            norm = np.linalg.norm(vector)

            if norm == 0:
                continue

            track_embeddings[mapped_idx] = vector / norm
            valid_tracks.append(mapped_idx)

        if not valid_tracks:
            self.use_hard_negatives = False
            return

        self.track_embeddings = track_embeddings
        self.valid_embedding_tracks = np.array(valid_tracks, dtype=np.int64)

    def sample_negatives(self, context, target, num_neg, rng):
        seen = set(context)
        seen.add(target)

        negatives = []

        if self.use_hard_negatives:
            desired_hard = int(round(num_neg * self.hard_negative_ratio))
            desired_hard = min(desired_hard, num_neg)

            hard_candidates = self.get_hard_negative_candidates(target)
            negatives.extend(
                self.take_candidates(
                    hard_candidates,
                    seen,
                    desired_hard,
                    rng,
                )
            )

        while len(negatives) < num_neg:
            candidate = rng.choice(self.all_tracks)

            if candidate in seen:
                continue

            negatives.append(candidate)
            seen.add(candidate)

        return negatives

    def take_candidates(self, candidates, seen, limit, rng):
        if limit <= 0 or not candidates:
            return []

        pool = list(candidates)
        rng.shuffle(pool)

        picked = []

        for candidate in pool:
            if candidate in seen:
                continue

            picked.append(candidate)
            seen.add(candidate)

            if len(picked) >= limit:
                break

        return picked

    def get_hard_negative_candidates(self, target):
        if not self.use_hard_negatives or self.track_embeddings is None:
            return []

        cached = self.hard_negative_cache.get(target)

        if cached is not None:
            return cached

        target_vec = self.track_embeddings[target]

        if not target_vec.any():
            self.hard_negative_cache[target] = []
            return []

        scores = self.track_embeddings[self.valid_embedding_tracks] @ target_vec
        top_k = min(self.hard_pool_size + 1, scores.shape[0])

        if top_k <= 1:
            self.hard_negative_cache[target] = []
            return []

        top_pos = np.argpartition(-scores, top_k - 1)[:top_k]
        ordered = top_pos[np.argsort(-scores[top_pos])]

        candidates = []

        for pos in ordered:
            candidate = int(self.valid_embedding_tracks[pos])

            if candidate == target:
                continue

            candidates.append(candidate)

            if len(candidates) >= self.hard_pool_size:
                break

        self.hard_negative_cache[target] = candidates
        return candidates


class RankingEvalDataset(HardNegativeMixin, IterableDataset):

    def __init__(
        self,
        db_path,
        track_map,
        num_candidates=100,
        max_context_len=MAX_CONTEXT_LEN,
        max_samples=None,
        seed=42,
        eval_mod=20,
        eval_remainder=0,
        embedding_file=TRACK_EMBEDDINGS_NPY,
        embedding_ids_file=TRACK_IDS_TXT,
        use_hard_negatives=True,
        hard_negative_ratio=0.7,
        hard_pool_size=128,
    ):
        self.db_path = db_path
        self.num_candidates = num_candidates
        self.max_context_len = max_context_len
        self.max_samples = max_samples
        self.seed = seed
        self.eval_mod = eval_mod
        self.eval_remainder = eval_remainder
        self.setup_negative_sampling(
            track_map=track_map,
            embedding_file=embedding_file,
            embedding_ids_file=embedding_ids_file,
            use_hard_negatives=use_hard_negatives,
            hard_negative_ratio=hard_negative_ratio,
            hard_pool_size=hard_pool_size,
        )

    def __iter__(self):
        con = duckdb.connect()
        con.execute("PRAGMA memory_limit='16GB'")

        rng = random.Random(self.seed)

        query = f"""
        SELECT playlist_rowid, track_rowid
        FROM read_parquet('{self.db_path}')
        WHERE track_rowid IS NOT NULL
        ORDER BY playlist_rowid, position
        """

        cursor = con.execute(query)

        current_playlist = []
        last_pid = None
        yielded = 0

        while True:
            batch = cursor.fetchmany(100000)

            if not batch:
                break

            for pid, track in batch:
                if track not in self.track_map:
                    continue

                track_id = self.track_map[track]

                if last_pid is not None and pid != last_pid:
                    sample = self.process_playlist(current_playlist, rng)

                    if sample is not None and self.is_eval_playlist(last_pid):
                        yield sample
                        yielded += 1

                        if self.max_samples is not None and yielded >= self.max_samples:
                            return

                    current_playlist = []

                current_playlist.append(track_id)
                last_pid = pid

        if len(current_playlist) > 1:
            sample = self.process_playlist(current_playlist, rng)

            if sample is not None and self.is_eval_playlist(last_pid):
                yield sample

    def is_eval_playlist(self, playlist_id):
        return playlist_id % self.eval_mod == self.eval_remainder

    def process_playlist(self, playlist, rng):
        if len(playlist) < 2:
            return None

        context = playlist[:-1][-self.max_context_len:]
        target = playlist[-1]

        if not context:
            return None

        negatives = self.sample_negatives(
            context=context,
            target=target,
            num_neg=max(0, self.num_candidates - 1),
            rng=rng,
        )

        candidates = [target] + negatives
        rng.shuffle(candidates)

        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(candidates, dtype=torch.long),
            torch.tensor(candidates.index(target), dtype=torch.long),
        )


class RankingDataset(HardNegativeMixin, IterableDataset):

    def __init__(
        self,
        db_path,
        track_map,
        num_neg=5,
        max_context_len=MAX_CONTEXT_LEN,
        seed=42,
        eval_mod=20,
        eval_remainder=0,
        embedding_file=TRACK_EMBEDDINGS_NPY,
        embedding_ids_file=TRACK_IDS_TXT,
        use_hard_negatives=True,
        hard_negative_ratio=0.4,
        hard_pool_size=128,
    ):
        self.db_path = db_path
        self.num_neg = num_neg
        self.max_context_len = max_context_len
        self.seed = seed
        self.eval_mod = eval_mod
        self.eval_remainder = eval_remainder
        self.setup_negative_sampling(
            track_map=track_map,
            embedding_file=embedding_file,
            embedding_ids_file=embedding_ids_file,
            use_hard_negatives=use_hard_negatives,
            hard_negative_ratio=hard_negative_ratio,
            hard_pool_size=hard_pool_size,
        )

    def __iter__(self):
        con = duckdb.connect()
        con.execute("PRAGMA memory_limit='16GB'")

        rng = random.Random(self.seed)

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
            batch = cursor.fetchmany(100000)

            if not batch:
                break

            for pid, track in batch:
                if track not in self.track_map:
                    continue

                track_id = self.track_map[track]

                if last_pid is not None and pid != last_pid:
                    if self.is_train_playlist(last_pid):
                        yield from self.process_playlist(current_playlist, rng)
                    current_playlist = []

                current_playlist.append(track_id)
                last_pid = pid

        if len(current_playlist) > 1 and self.is_train_playlist(last_pid):
            yield from self.process_playlist(current_playlist, rng)

    def is_train_playlist(self, playlist_id):
        return playlist_id % self.eval_mod != self.eval_remainder

    def process_playlist(self, playlist, rng):
        if len(playlist) < 2:
            return

        for i in range(1, len(playlist)):
            context = playlist[:i][-self.max_context_len:]
            target = playlist[i]

            yield (
                torch.tensor(context, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
                torch.tensor(1.0, dtype=torch.float32),
            )

            negatives = self.sample_negatives(
                context=context,
                target=target,
                num_neg=self.num_neg,
                rng=rng,
            )

            for neg in negatives:
                yield (
                    torch.tensor(context, dtype=torch.long),
                    torch.tensor(neg, dtype=torch.long),
                    torch.tensor(0.0, dtype=torch.float32),
                )
