from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

EXTERNAL_DB_DIR = Path(r"D:\ZZZFORALL\dbmusic")


def _prefer_local_raw(filename: str) -> Path:
    local_path = RAW_DATA_DIR / filename
    if local_path.exists():
        return local_path
    return EXTERNAL_DB_DIR / filename


TRACKS_PARQUET = _prefer_local_raw("tracks.parquet")
PLAYLIST_TRACKS_PARQUET = _prefer_local_raw("playlist_tracks.parquet")
TRACK_AUDIO_FEATURES_PARQUET = _prefer_local_raw("track_audio_features.parquet")
TRACK_ARTISTS_PARQUET = _prefer_local_raw("track_artists.parquet")
ARTISTS_PARQUET = _prefer_local_raw("artists.parquet")
ARTIST_GENRES_PARQUET = _prefer_local_raw("artist_genres.parquet")
ALBUMS_PARQUET = _prefer_local_raw("albums.parquet")
TRACK_FEATURE_STORE_PARQUET = _prefer_local_raw("track_feature_store.parquet")

TOP_TRACKS_PARQUET = PROCESSED_DATA_DIR / "top_tracks.parquet"
PLAYLIST_TRACKS_FILTERED_PARQUET = PROCESSED_DATA_DIR / "playlist_tracks_filtered.parquet"
AUDIO_FEATURES_FILTERED_PARQUET = PROCESSED_DATA_DIR / "audio_features_filtered.parquet"
AUDIO_INDEX_PKL = PROCESSED_DATA_DIR / "audio_index.pkl"
TRACK_ID_MAP_PKL = PROCESSED_DATA_DIR / "track_id_map.pkl"
TRACK_SEARCH_INDEX_PARQUET = PROCESSED_DATA_DIR / "track_search_index.parquet"
FILTERED_TRACK_ID_MAP_PKL = TRACK_ID_MAP_PKL
RANKING_MODEL_PT = PROCESSED_DATA_DIR / "ranking_model.pt"
TWO_TOWER_MODEL_PT = PROCESSED_DATA_DIR / "two_tower.pt"
TRANSFORMER_MODEL_PT = PROCESSED_DATA_DIR / "transformer_model.pt"

TRACK2VEC_MODEL = EMBEDDINGS_DIR / "track2vec.model"
TRACK2VEC_SYN1NEG = EMBEDDINGS_DIR / "track2vec.model.syn1neg.npy"
TRACK2VEC_VECTORS = EMBEDDINGS_DIR / "track2vec.model.wv.vectors.npy"
TRACK_EMBEDDINGS_NPY = EMBEDDINGS_DIR / "track_embeddings.npy"
TRACK_IDS_TXT = EMBEDDINGS_DIR / "track_ids.txt"
TRACK2VEC_TRACK_MAP_PKL = EMBEDDINGS_DIR / "track2vec_track_id_map.pkl"

EMBED_DIM = 512
BATCH_SIZE = 256
SEQ_LEN = 20
LR = 1e-3
EPOCHS = 1
