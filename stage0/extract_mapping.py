import sys
from pathlib import Path

from gensim.models import Word2Vec
import pickle

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import TRACK2VEC_MODEL, TRACK2VEC_TRACK_MAP_PKL

print("Loading model...")

model = Word2Vec.load(str(TRACK2VEC_MODEL))

print("Extracting mapping...")

mapping = {}

for k, v in model.wv.key_to_index.items():
    mapping[int(k)] = v

print("Saving...")

with open(TRACK2VEC_TRACK_MAP_PKL, "wb") as f:
    pickle.dump(mapping, f)

print("Done.")
