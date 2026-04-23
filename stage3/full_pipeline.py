from stage1.candidate_generator import generate_stage1_candidates
from stage2.model_infer import score_sequence_model  # 👈 добавим
from stage3.pipeline import rerank_candidates


def build_features(
    playlist_track_ids,
    candidate_ids,
    stage1_scores,
    transformer_scores,
    popularity_map=None,
):
    features = []

    for i, track_id in enumerate(candidate_ids):
        feat = {
            "track_id": track_id,
            "stage1_score": stage1_scores.get(track_id, 0.0),
            "transformer_score": transformer_scores.get(track_id, 0.0),
            "popularity": popularity_map.get(track_id, 0.0) if popularity_map else 0.0,
        }
        features.append(feat)

    return features


def postprocess(recommendations, playlist_track_ids, top_k=50):
    seen = set(playlist_track_ids)

    filtered = [r for r in recommendations if r["track_id"] not in seen]

    # простая диверсификация (можно усложнить)
    return filtered[:top_k]


def run_full_pipeline(
    playlist_track_ids,
    stage1_assets,
    stage2_assets,
    stage3_assets,
    top_k_stage1=300,
):
    # ---- Stage 1 ----
    candidates = generate_stage1_candidates(
        playlist_ids=playlist_track_ids,
        top_k=top_k_stage1,
        assets=stage1_assets,
    )

    candidate_ids = [c["track_id"] for c in candidates]
    stage1_scores = {c["track_id"]: c["score"] for c in candidates}

    # ---- Stage 2 (Transformer scoring) ----
    transformer_scores = score_sequence_model(
        model=stage2_assets["model"],
        playlist_track_ids=playlist_track_ids,
        candidate_track_ids=candidate_ids,
        track_map=stage2_assets["track_map"],
        pad_idx=stage2_assets["pad_idx"],
        device=stage2_assets["device"],
    )

    # ---- Feature building ----
    features = build_features(
        playlist_track_ids,
        candidate_ids,
        stage1_scores,
        transformer_scores,
        popularity_map=stage3_assets.get("popularity_map"),
    )

    # ---- Stage 3 ranking ----
    ranked = rerank_candidates(
        model=stage3_assets["model"],
        features=features,  # 👈 ВАЖНО: теперь передаем фичи
        device=stage3_assets["device"],
    )

    # ---- Postprocess ----
    final = postprocess(ranked, playlist_track_ids)

    return final
