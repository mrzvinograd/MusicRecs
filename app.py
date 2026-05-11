import streamlit as st

from config import STAGE1_TRACK_VECTORS_NPY, TRACK_SEARCH_INDEX_PARQUET, TRANSFORMER_MODEL_PT
from stage1.candidate_generator import generate_stage1_candidates, load_stage1_assets
from stage2.model_infer import score_sequence_model
from stage2.pipeline import load_stage2_assets
from stage3.pipeline import load_stage3_assets, rerank_candidates
from utils.music_metadata import fetch_track_metadata, resolve_track_token


st.set_page_config(page_title="MusicRecs", layout="wide")


@st.cache_resource
def get_stage1_assets():
    return load_stage1_assets()


@st.cache_resource
def get_stage2_assets():
    if not TRANSFORMER_MODEL_PT.exists():
        return None, f"Missing checkpoint: {TRANSFORMER_MODEL_PT}"

    try:
        return load_stage2_assets(), None
    except Exception as exc:  # pragma: no cover - UI fallback
        return None, str(exc)


@st.cache_resource
def get_stage3_assets():
    return load_stage3_assets()


def ensure_state():
    st.session_state.setdefault("playlist_ids", [])


def add_track(track_id):
    if track_id not in st.session_state.playlist_ids:
        st.session_state.playlist_ids.append(track_id)


def remove_track(track_id):
    st.session_state.playlist_ids = [item for item in st.session_state.playlist_ids if item != track_id]


def replace_playlist_from_text(raw_text):
    parsed = []

    for chunk in raw_text.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parsed.append(resolve_track_token(chunk))

    st.session_state.playlist_ids = parsed


def render_playlist_editor():
    st.subheader("Входной плейлист")
    st.caption("Вставьте Spotify track ID, например `7AB0cUXnzuSlAnyHOqmrZr`: по одному в строке или через запятую.")

    if not TRACK_SEARCH_INDEX_PARQUET.exists():
        st.warning(
            "Индекс Spotify ID не найден. Соберите его один раз командой "
            "`python scripts/build_track_search_index.py`, затем перезапустите приложение."
        )

    existing_metadata = fetch_track_metadata(st.session_state.playlist_ids) if st.session_state.playlist_ids else {}
    current_spotify_ids = [
        existing_metadata.get(track_id, {}).get("spotify_track_id", str(track_id))
        for track_id in st.session_state.playlist_ids
    ]

    with st.expander("Вставить Spotify Track ID", expanded=True):
        raw_ids = st.text_area(
            "Spotify track IDs",
            value="\n".join(current_spotify_ids),
            height=120,
            help="Use Spotify track IDs only. Example: 7AB0cUXnzuSlAnyHOqmrZr",
        )
        if st.button("Применить ID"):
            try:
                replace_playlist_from_text(raw_ids)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()

    playlist_ids = st.session_state.playlist_ids
    st.caption(f"Выбрано треков: {len(playlist_ids)}")

    if not playlist_ids:
        st.warning("Добавьте хотя бы один трек, чтобы получить рекомендации.")
        return playlist_ids

    metadata = existing_metadata if playlist_ids else {}

    for index, track_id in enumerate(playlist_ids, start=1):
        meta = metadata.get(track_id, {})
        cols = st.columns([8, 1])
        with cols[0]:
            st.write(
                f"{index}. **{meta.get('track_name', 'Неизвестный трек')}** - "
                f"{meta.get('artist_names', 'Неизвестный исполнитель')}  \n"
                f"`track_rowid={track_id}`  `spotify_id={meta.get('spotify_track_id', 'unknown')}`"
            )
        with cols[1]:
            if st.button("Удалить", key=f"remove_{track_id}"):
                remove_track(track_id)
                st.rerun()

    return playlist_ids


def enrich_items(track_ids, score_lookup):
    metadata = fetch_track_metadata(track_ids)
    rows = []

    for rank, track_id in enumerate(track_ids, start=1):
        meta = metadata.get(track_id, {})
        rows.append(
            {
                "rank": rank,
                "track_id": track_id,
                "score": float(score_lookup.get(track_id, 0.0)),
                "track_name": meta.get("track_name", "Неизвестный трек"),
                "artist_names": meta.get("artist_names", "Неизвестный исполнитель"),
                "album_name": meta.get("album_name", "Неизвестный альбом"),
                "spotify_track_id": meta.get("spotify_track_id", "Неизвестный Spotify ID"),
            }
        )

    return rows


def run_pipeline(playlist_track_ids, candidate_k, stage2_k, top_k):
    stage1_assets = get_stage1_assets()
    stage2_assets, stage2_error = get_stage2_assets()
    stage3_assets = get_stage3_assets()

    stage1_candidates = generate_stage1_candidates(
        playlist_ids=playlist_track_ids,
        top_k=candidate_k,
        assets=stage1_assets,
    )
    stage1_score_lookup = {item["track_id"]: item["score"] for item in stage1_candidates}
    stage1_ids = [item["track_id"] for item in stage1_candidates]

    stage2_available = stage2_assets is not None

    if stage2_available:
        stage2_scores = score_sequence_model(
            model=stage2_assets["model"],
            playlist_track_ids=playlist_track_ids,
            candidate_track_ids=stage1_ids,
            track_map=stage2_assets["track_map"],
            pad_idx=stage2_assets["pad_idx"],
            device=stage2_assets["device"],
            max_len=stage2_assets["seq_len"],
        )
        stage2_ids = sorted(stage1_ids, key=lambda track_id: stage2_scores.get(track_id, 0.0), reverse=True)[:stage2_k]
    else:
        stage2_scores = {}
        stage2_ids = stage1_ids[:stage2_k]

    final_ranked = rerank_candidates(
        model=stage3_assets["model"],
        playlist_track_ids=playlist_track_ids,
        candidate_track_ids=stage2_ids,
        filtered_track_map=stage3_assets["track_map"],
        pad_idx=stage3_assets["pad_idx"],
        device=stage3_assets["device"],
    )
    final_ids = [item["track_id"] for item in final_ranked[:top_k]]
    final_score_lookup = {item["track_id"]: item["score"] for item in final_ranked}

    return {
        "stage2_enabled": stage2_available,
        "stage2_error": stage2_error,
        "stage1_top": enrich_items(stage1_ids[:top_k], stage1_score_lookup),
        "stage2_top": enrich_items(stage2_ids[:top_k], stage2_scores if stage2_available else stage1_score_lookup),
        "final_top": enrich_items(final_ids, final_score_lookup),
    }


def main():
    ensure_state()

    st.title("MusicRecs")
    st.write("Соберите плейлист из Spotify track ID и запустите рекомендательную систему.")

    if not STAGE1_TRACK_VECTORS_NPY.exists():
        st.info(
            "Для быстрых рекомендаций соберите кеш stage1 один раз: "
            "`python scripts/build_stage1_track_vectors.py`. Без кеша первый этап будет считать весь каталог заново."
        )

    playlist_ids = render_playlist_editor()

    col1, col2, col3 = st.columns(3)
    with col1:
        candidate_k = st.number_input(
            "Кандидатов Stage1",
            min_value=50,
            max_value=1000,
            value=300,
            step=50,
            help="Сколько первичных кандидатов stage1 берёт из каталога перед последующей сортировкой.",
        )
    with col2:
        stage2_k = st.number_input(
            "Оставить после Stage2",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Сколько кандидатов после stage2 передаётся в stage3.",
        )
    with col3:
        top_k = st.number_input(
            "Показать top-k",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Сколько финальных рекомендаций показать после полного пайплайна.",
        )

    if st.button("Получить рекомендации", type="primary", disabled=not playlist_ids):
        with st.spinner("Запускаю stage1/stage2/stage3..."):
            result = run_pipeline(
                playlist_track_ids=playlist_ids,
                candidate_k=int(candidate_k),
                stage2_k=int(stage2_k),
                top_k=int(top_k),
            )

        st.write(f"Stage2 включён: {'да' if result['stage2_enabled'] else 'нет'}")
        if result["stage2_error"]:
            st.info(f"Примечание stage2: {result['stage2_error']}")

        tabs = st.tabs(["Stage1", "Stage2", "Финальные рекомендации"])
        with tabs[0]:
            st.dataframe(result["stage1_top"], use_container_width=True)
        with tabs[1]:
            st.dataframe(result["stage2_top"], use_container_width=True)
        with tabs[2]:
            st.dataframe(result["final_top"], use_container_width=True)


if __name__ == "__main__":
    main()
