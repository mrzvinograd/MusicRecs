import sys
from pathlib import Path

from gensim.models import Word2Vec
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import TRACK2VEC_MODEL, TRACK_EMBEDDINGS_NPY, TRACK_IDS_TXT


model = Word2Vec.load(str(TRACK2VEC_MODEL))

vectors = model.wv.vectors
ids = model.wv.index_to_key

np.save(TRACK_EMBEDDINGS_NPY, vectors)

with open(TRACK_IDS_TXT, "w", encoding="utf-8") as f:
    for t in ids:
        f.write(t + "\n")

print("Embeddings exported")
