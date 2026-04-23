# zsoundproject

Music recommendation project organized as a multi-stage pipeline.

## Structure

- `data/raw/`
  Raw local data placeholders. The main Spotify parquet sources currently live in `D:\ZZZFORALL\dbmusic`.
- `data/processed/`
  Derived parquet and checkpoint artifacts such as filtered playlists, maps, and trained model weights.
- `data/embeddings/`
  `track2vec` model files and exported embedding artifacts.
- `stage0/`
  Word2Vec / `track2vec` training and embedding export utilities.
- `stage1/`
  Two-tower retrieval training, dataset code, and models.
- `stage2/`
  Transformer next-track training, dataset code, and models.
- `stage3/`
  Ranking dataset, training, evaluation, inference, and models.
- `scripts/`
  Data preparation scripts for filtered parquet files, maps, and feature indices.
- `utils/`
  Shared helpers area for future common logic.
- `config.py`
  Centralized paths and shared constants.

## Main entrypoints

- Stage 0 training:
  `python stage0/train_track2vec.py`
- Stage 0 map export:
  `python stage0/extract_mapping.py`
- Stage 0 artifact validation:
  `python stage0/validate_artifacts.py`
- Stage 1 training:
  `python stage1/train_two_tower.py`
- Stage 1 candidate generation:
  `python stage1/recommend_candidates.py --playlist "10000,1178,2779"`
- Stage 2 training:
  `python stage2/train_transformer.py`
- Stage 3 training:
  `python stage3/train_ranking.py`
- Stage 3 evaluation:
  `python stage3/eval_ranking.py`
- Stage 3 recommendations:
  `python stage3/recommend_tracks.py --playlist "10000,1178,2779"`
- End-to-end pipeline recommendations:
  `python stage3/recommend_pipeline.py --playlist "10000,1178,2779"`
- End-to-end pipeline evaluation:
  `python stage3/eval_pipeline.py`

## Notes

- Shared paths are defined in [config.py](/D:/ZZZFORALL/zsoundproject/config.py).
- The external raw Spotify parquet files are still referenced from `D:\ZZZFORALL\dbmusic`.
- Trained artifacts are now stored under `data/processed/` and `data/embeddings/`.
