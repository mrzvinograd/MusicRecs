import streamlit as st

from config import TRACK_SEARCH_INDEX_PARQUET, TRANSFORMER_MODEL_PT
from stage1.candidate_generator import generate_stage1_candidates, load_stage1_assets
from stage2.model_infer import score_sequence_model
from stage2.pipeline import load_stage2_assets
from stage3.pipeline import load_stage3_assets, rerank_candidates
from utils.music_metadata import fetch_track_metadata, resolve_track_token, search_tracks


st.set_page_config(page_title="zsoundproject", layout="wide")


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
    st.subheader("Playlist Input")
    st.caption("Search by track, artist, or album. Manual input supports numeric `track_rowid` and Spotify track IDs from the dataset.")

    if not TRACK_SEARCH_INDEX_PARQUET.exists():
        st.warning(
            "Track search index is missing. Build it once with "
            "`python scripts/build_track_search_index.py`, then restart Streamlit."
        )

    query = st.text_input("Search tracks", placeholder="Track, artist, album")

    if query and len(query.strip()) >= 2 and TRACK_SEARCH_INDEX_PARQUET.exists():
        try:
            results = search_tracks(query, limit=10)
        except Exception as exc:
            st.error(f"Track search failed: {exc}")
            results = []

        if results:
            for item in results:
                cols = st.columns([8, 1])
                with cols[0]:
                    st.write(
                        f"**{item['track_name']}** - {item['artist_names']} "
                        f"`[{item['track_id']}]`  \nAlbum: {item['album_name']}"
                    )
                with cols[1]:
                    if st.button("Add", key=f"add_{item['track_id']}"):
                        add_track(item["track_id"])
                        st.rerun()
        else:
            st.info("No matches found.")
    elif query:
        st.caption("Type at least 2 characters to search.")

    with st.expander("Or paste track IDs"):
        raw_ids = st.text_area(
            "Comma-separated or one per line",
            value=",".join(str(track_id) for track_id in st.session_state.playlist_ids),
            height=120,
        )
        if st.button("Apply IDs"):
            try:
                replace_playlist_from_text(raw_ids)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()

    playlist_ids = st.session_state.playlist_ids
    st.caption(f"Selected tracks: {len(playlist_ids)}")

    if not playlist_ids:
        st.warning("Add at least one track to build recommendations.")
        return playlist_ids

    metadata = fetch_track_metadata(playlist_ids) if TRACK_SEARCH_INDEX_PARQUET.exists() else {}

    for index, track_id in enumerate(playlist_ids, start=1):
        meta = metadata.get(track_id, {})
        cols = st.columns([8, 1])
        with cols[0]:
            st.write(
                f"{index}. **{meta.get('track_name', 'Unknown Track')}** - "
                f"{meta.get('artist_names', 'Unknown Artist')} "
                f"`[{track_id}]`"
            )
        with cols[1]:
            if st.button("Remove", key=f"remove_{track_id}"):
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
                "track_name": meta.get("track_name", "Unknown Track"),
                "artist_names": meta.get("artist_names", "Unknown Artist"),
                "album_name": meta.get("album_name", "Unknown Album"),
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

    st.title("zsoundproject MVP")
    st.write("Build a playlist from track IDs or search results, then run the full recommendation pipeline.")

    playlist_ids = render_playlist_editor()

    col1, col2, col3 = st.columns(3)
    with col1:
        candidate_k = st.number_input("Stage1 candidate-k", min_value=50, max_value=1000, value=300, step=50)
    with col2:
        stage2_k = st.number_input("Stage2 keep-k", min_value=10, max_value=500, value=100, step=10)
    with col3:
        top_k = st.number_input("Display top-k", min_value=5, max_value=100, value=20, step=5)

    if st.button("Run Recommendations", type="primary", disabled=not playlist_ids):
        with st.spinner("Running stage1/stage2/stage3..."):
            result = run_pipeline(
                playlist_track_ids=playlist_ids,
                candidate_k=int(candidate_k),
                stage2_k=int(stage2_k),
                top_k=int(top_k),
            )

        st.write(f"Stage2 enabled: {'yes' if result['stage2_enabled'] else 'no'}")
        if result["stage2_error"]:
            st.info(f"Stage2 note: {result['stage2_error']}")

        tabs = st.tabs(["Stage1", "Stage2", "Final"])
        with tabs[0]:
            st.dataframe(result["stage1_top"], use_container_width=True)
        with tabs[1]:
            st.dataframe(result["stage2_top"], use_container_width=True)
        with tabs[2]:
            st.dataframe(result["final_top"], use_container_width=True)


if __name__ == "__main__":
    main()
