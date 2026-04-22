import sys
from pathlib import Path
import pickle

import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import AUDIO_FEATURES_FILTERED_PARQUET, AUDIO_INDEX_PKL


print("Loading filtered audio...")

audio = pl.read_parquet(AUDIO_FEATURES_FILTERED_PARQUET)

print("Building index...")

audio_index = {}

for row in audio.iter_rows(named=True):
    audio_index[row["track_id"]] = [
        row["danceability"],
        row["energy"],
        row["tempo"],
        row["loudness"],
        row["speechiness"],
        row["acousticness"],
        row["instrumentalness"],
        row["liveness"],
        row["valence"],
    ]

print("Saving...")

with open(AUDIO_INDEX_PKL, "wb") as f:
    pickle.dump(audio_index, f)

print(f"Done. Saved to {AUDIO_INDEX_PKL}")
