import sys
from pathlib import Path

from gensim.models import Word2Vec

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import TRACK2VEC_MODEL


model = Word2Vec.load(str(TRACK2VEC_MODEL))

print("Vocabulary size:", len(model.wv))

track = model.wv.index_to_key[0]

print("\nTest track:", track)

similar = model.wv.most_similar(track, topn=10)

print("\nMost similar tracks:")

for t, score in similar:
    print(t, score)
