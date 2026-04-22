import sys
from pathlib import Path

import pickle

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FILTERED_TRACK_ID_MAP_PKL, PLAYLIST_TRACKS_PARQUET, RANKING_MODEL_PT
from stage3.evaluate_ranking import evaluate_ranking
from stage3.models.model_ranking import RankingModel


DB_PATH = str(PLAYLIST_TRACKS_PARQUET)
MODEL_PATH = RANKING_MODEL_PT
EVAL_MOD = 20
EVAL_REMAINDER = 0
USE_HARD_NEGATIVES = True
EVAL_HARD_NEGATIVE_RATIO = 0.7
HARD_NEGATIVE_POOL_SIZE = 128


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


with open(FILTERED_TRACK_ID_MAP_PKL, "rb") as f:
    track_map = pickle.load(f)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model_kwargs = checkpoint.get(
    "model_kwargs",
    {
        "vocab_size": checkpoint["vocab_size"],
        "padding_idx": checkpoint["pad_idx"],
    },
)

model = RankingModel(**model_kwargs).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

metrics = evaluate_ranking(
    model=model,
    db_path=DB_PATH,
    track_map=track_map,
    pad_idx=model_kwargs["padding_idx"],
    k_values=(10, 50),
    num_candidates=100,
    num_samples=2000,
    batch_size=32,
    eval_mod=EVAL_MOD,
    eval_remainder=EVAL_REMAINDER,
    use_hard_negatives=USE_HARD_NEGATIVES,
    hard_negative_ratio=EVAL_HARD_NEGATIVE_RATIO,
    hard_pool_size=HARD_NEGATIVE_POOL_SIZE,
)

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
