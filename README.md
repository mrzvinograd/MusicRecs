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
- Filter audio features for top tracks:
  `python scripts/build_filtered_audio.py`
- Build normalized audio feature index:
  `python scripts/prepare_feature_indices.py`
- Track search index build:
  `python scripts/build_track_search_index.py`
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
- End-to-end system evaluation:
  `python stage3/eval_system.py --samples 50 --candidate-k 300 --stage2-k 100`
- End-to-end system recommendations:
  `python stage3/recommend_system.py --playlist "10000,1178,2779" --top-k 20`
- Streamlit UI:
  `streamlit run app.py`

## Notes

- Shared paths are defined in [config.py](/D:/ZZZFORALL/zsoundproject/config.py).
- The external raw Spotify parquet files are still referenced from `D:\ZZZFORALL\dbmusic`.
- Trained artifacts are now stored under `data/processed/` and `data/embeddings/`.
- The Streamlit app expects `data/processed/track_search_index.parquet`; build it once with `python scripts/build_track_search_index.py`.
- `stage1` now uses normalized audio features from `track_audio_features.parquet` when `audio_index.pkl` exists.

## UI Parameters

- `Stage1 candidate-k`
  How many candidate tracks the retrieval model takes from the full catalog before any later reranking. Higher values usually improve recall but make later stages slower.
- `Stage2 keep-k`
  How many of the stage1 candidates survive after stage2 sequence scoring and get passed to stage3. This is the bridge between recall and precision.
- `Display top-k`
  How many final recommendations are shown to the user after the whole pipeline finishes. This changes presentation, not training.

## Clean-Run Full Retrain Checklist

Run this from repo root for a full retrain without jumping back between stages.

0. Optional cleanup of generated artifacts before a fresh run:
   `Remove-Item -Force data\\processed\\* -ErrorAction SilentlyContinue`
   `Remove-Item -Force data\\embeddings\\* -ErrorAction SilentlyContinue`
1. Build top-track frequency base:
   `python scripts/build_track_frequency.py`
2. Freeze the filtered training universe:
   `python scripts/build_filtered_sessions.py`
3. Build the filtered track id map used by stages 2 and 3:
   `python scripts/build_track_map.py`
4. Export filtered audio features for the same universe:
   `python scripts/build_filtered_audio.py`
5. Build normalized audio arrays consumed by stage1:
   `python scripts/prepare_feature_indices.py`
6. Train `track2vec` on filtered playlists:
   `python stage0/train_track2vec.py`
7. Export stage0 embeddings:
   `python stage0/export_embeddings.py`
8. Export the stage0 track map used by stage1:
   `python stage0/extract_mapping.py`
9. Validate stage0 artifacts before later stages:
   `python stage0/validate_artifacts.py`
10. Build the Spotify-id search index for UI and metadata lookup:
   `python scripts/build_track_search_index.py`
11. Train stage1 retrieval:
   `python stage1/train_two_tower.py`
12. Train stage2 next-track transformer:
   `python stage2/train_transformer.py`
13. Train stage3 reranker:
   `python stage3/train_ranking.py`
14. Run end-to-end evaluation:
   `python stage3/eval_system.py --samples 50 --candidate-k 300 --stage2-k 100`
15. Smoke-test the final system:
   `python stage3/recommend_system.py --playlist "10000,1178,2779" --top-k 20`
16. Launch the UI:
   `streamlit run app.py`

Artifacts by stage:
- Steps `1-5` prepare `data/processed/` inputs for all later stages.
- Steps `6-9` prepare `data/embeddings/` plus the stage1 map.
- Step `10` is only for metadata lookup and the Streamlit app, so keep it after the dataset universe is finalized.
- Steps `11-13` train the three model stages in dependency order.

