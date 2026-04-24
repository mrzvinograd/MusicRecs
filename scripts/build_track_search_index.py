import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import TRACK_SEARCH_INDEX_PARQUET
from utils.music_metadata import ensure_track_search_index


print("Building track search index...")
path = ensure_track_search_index()
print(f"Done. Saved to {path}")
