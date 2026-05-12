import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from config import PLAYLIST_TRACKS_PARQUET, STAGE1_TRACK_VECTORS_NPY, TRANSFORMER_MODEL_PT
from stage1.candidate_generator import generate_stage1_candidates, load_stage1_assets
from stage2.model_infer import score_sequence_model
from stage2.pipeline import load_stage2_assets
from stage3.eval_pipeline import DUCKDB_MEMORY_LIMIT, stream_eval_playlists
from stage3.pipeline import load_stage3_assets, rerank_candidates


DEFAULT_K_VALUES = (10, 50)
EVALUATION_MEMORY_LIMIT = DUCKDB_MEMORY_LIMIT
REPORT_FIELDS = (
    "stage",
    "k",
    "recall",
    "ndcg",
    "mrr",
    "hits",
    "samples",
    "playlist_embedding_similarity",
    "target_embedding_similarity",
    "playlist_audio_similarity",
    "random_playlist_embedding_similarity",
    "random_target_embedding_similarity",
    "random_playlist_audio_similarity",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Оценка качества рекомендаций MusicRecs по метрикам Recall@K, "
            "NDCG@K и MRR@K для Stage1, Stage2 и финального пайплайна."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Количество контрольных плейлистов для оценки.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=300,
        help="Сколько кандидатов Stage1 генерирует перед последующей пересортировкой.",
    )
    parser.add_argument(
        "--stage2-k",
        type=int,
        default=100,
        help="Сколько кандидатов оставить после Stage2 перед финальным Stage3 reranking.",
    )
    parser.add_argument(
        "--k-values",
        default="10,50",
        help="Значения K для метрик через запятую, например: 10,50.",
    )
    parser.add_argument(
        "--eval-mod",
        type=int,
        default=20,
        help=(
            "Модуль разбиения плейлистов. Плейлист попадает в оценку, "
            "если playlist_rowid %% eval_mod == eval_remainder."
        ),
    )
    parser.add_argument(
        "--eval-remainder",
        type=int,
        default=0,
        help="Остаток от деления для выбора контрольной выборки.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Необязательный путь для сохранения отчёта. Поддерживаются расширения .csv и .json.",
    )
    return parser.parse_args()


def parse_k_values(raw_value):
    # Метрики считаются сразу для нескольких K, например Recall@10 и Recall@50.
    # Set убирает дубли, sorted делает порядок вывода стабильным для отчёта.
    values = sorted({int(item.strip()) for item in raw_value.split(",") if item.strip()})
    return tuple(values) or DEFAULT_K_VALUES


def new_metrics(k_values):
    # Для каждого K храним количество попаданий и накопленные суммы NDCG/MRR.
    # Деление на число примеров выполняется в конце, когда известен total.
    return {k: {"hits": 0, "ndcg": 0.0, "mrr": 0.0} for k in k_values}


def new_similarity_metrics(k_values):
    # Эти метрики не проверяют точное совпадение со скрытым следующим треком.
    # Они измеряют музыкальную близость рекомендаций к плейлисту и target-треку,
    # что лучше отражает задачу "передать вайб" исходного плейлиста.
    metric_names = (
        "playlist_embedding_similarity",
        "target_embedding_similarity",
        "playlist_audio_similarity",
        "random_playlist_embedding_similarity",
        "random_target_embedding_similarity",
        "random_playlist_audio_similarity",
    )
    return {
        k: {name: {"sum": 0.0, "count": 0} for name in metric_names}
        for k in k_values
    }


def update_metrics(metrics, ranked_track_ids, target_track_id, k_values):
    # В качестве релевантного элемента используется последний трек плейлиста.
    # Модель получает предыдущие треки и должна поставить скрытый target выше.
    for k in k_values:
        topk = ranked_track_ids[:k]

        if target_track_id not in topk:
            continue

        rank = topk.index(target_track_id) + 1
        metrics[k]["hits"] += 1
        # Recall@K равен 1 для примера, если целевой трек попал в top-K.
        # NDCG@K и MRR@K дополнительно учитывают позицию найденного трека.
        metrics[k]["ndcg"] += 1.0 / math.log2(rank + 1)
        metrics[k]["mrr"] += 1.0 / rank


def _append_similarity(metrics, k, name, value):
    if value is None or not np.isfinite(value):
        return

    metrics[k][name]["sum"] += float(value)
    metrics[k][name]["count"] += 1


def _track_indices(track_ids, track_map):
    valid_track_ids = []
    indices = []

    for track_id in track_ids:
        mapped_idx = track_map.get(track_id)

        if mapped_idx is None:
            continue

        valid_track_ids.append(track_id)
        indices.append(mapped_idx)

    return valid_track_ids, np.asarray(indices, dtype=np.int64)


def _feature_matrix(track_ids, stage1_assets, feature_kind):
    # Для оценки "вайба" используем уже подготовленные признаки Stage1:
    # track2vec-эмбеддинги и нормализованные аудио-признаки треков.
    valid_track_ids, indices = _track_indices(track_ids, stage1_assets["track_map"])

    if indices.size == 0:
        return None

    if feature_kind == "embedding":
        return np.asarray(stage1_assets["embeddings"][indices], dtype=np.float32)

    if feature_kind != "audio":
        raise ValueError(f"Неизвестный тип признаков: {feature_kind}")

    audio_features = stage1_assets["audio_features"]

    if audio_features is None:
        return None

    if hasattr(audio_features, "features_for_track_ids"):
        return audio_features.features_for_track_ids(valid_track_ids)

    return np.asarray(audio_features[indices], dtype=np.float32)


def _normalize_matrix(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _similarity_to_playlist(recommended_ids, playlist_ids, stage1_assets, feature_kind):
    recommended = _feature_matrix(recommended_ids, stage1_assets, feature_kind)
    playlist = _feature_matrix(playlist_ids, stage1_assets, feature_kind)

    if recommended is None or playlist is None:
        return None

    playlist_centroid = playlist.mean(axis=0, keepdims=True)
    playlist_centroid = _normalize_matrix(playlist_centroid)[0]
    recommended = _normalize_matrix(recommended)
    return float(np.mean(recommended @ playlist_centroid))


def _similarity_to_target(recommended_ids, target_track_id, stage1_assets, feature_kind):
    recommended = _feature_matrix(recommended_ids, stage1_assets, feature_kind)
    target = _feature_matrix([target_track_id], stage1_assets, feature_kind)

    if recommended is None or target is None:
        return None

    recommended = _normalize_matrix(recommended)
    target = _normalize_matrix(target)[0]
    return float(np.max(recommended @ target))


def sample_random_track_ids(stage1_assets, excluded_track_ids, count, rng):
    # Random baseline нужен для отчёта: он показывает, что рекомендации похожи
    # на плейлист сильнее, чем случайные треки из того же каталога.
    reverse_track_ids = stage1_assets["reverse_track_ids"]
    excluded_track_ids = set(excluded_track_ids)
    sampled = []
    attempts = 0
    max_attempts = max(count * 50, 1000)

    while len(sampled) < count and attempts < max_attempts:
        attempts += 1
        track_id = int(reverse_track_ids[int(rng.integers(0, reverse_track_ids.shape[0]))])

        if track_id in excluded_track_ids:
            continue

        sampled.append(track_id)
        excluded_track_ids.add(track_id)

    return sampled


def update_similarity_metrics(
    metrics,
    ranked_track_ids,
    random_track_ids,
    playlist_track_ids,
    target_track_id,
    stage1_assets,
    k_values,
):
    for k in k_values:
        recommended_topk = ranked_track_ids[:k]
        random_topk = random_track_ids[:k]

        _append_similarity(
            metrics,
            k,
            "playlist_embedding_similarity",
            _similarity_to_playlist(recommended_topk, playlist_track_ids, stage1_assets, "embedding"),
        )
        _append_similarity(
            metrics,
            k,
            "target_embedding_similarity",
            _similarity_to_target(recommended_topk, target_track_id, stage1_assets, "embedding"),
        )
        _append_similarity(
            metrics,
            k,
            "playlist_audio_similarity",
            _similarity_to_playlist(recommended_topk, playlist_track_ids, stage1_assets, "audio"),
        )
        _append_similarity(
            metrics,
            k,
            "random_playlist_embedding_similarity",
            _similarity_to_playlist(random_topk, playlist_track_ids, stage1_assets, "embedding"),
        )
        _append_similarity(
            metrics,
            k,
            "random_target_embedding_similarity",
            _similarity_to_target(random_topk, target_track_id, stage1_assets, "embedding"),
        )
        _append_similarity(
            metrics,
            k,
            "random_playlist_audio_similarity",
            _similarity_to_playlist(random_topk, playlist_track_ids, stage1_assets, "audio"),
        )


def _mean_similarity(similarity_metrics, k, name):
    item = similarity_metrics[k][name]

    if item["count"] == 0:
        return None

    return item["sum"] / item["count"]


def flatten_metrics(stage_name, metrics, similarity_metrics, total, k_values):
    # Преобразуем накопленные значения в табличный формат, который удобно
    # вставлять в отчёт ВКР или сохранять в CSV/JSON.
    rows = []

    for k in k_values:
        rows.append(
            {
                "stage": stage_name,
                "k": k,
                "recall": metrics[k]["hits"] / total if total else 0.0,
                "ndcg": metrics[k]["ndcg"] / total if total else 0.0,
                "mrr": metrics[k]["mrr"] / total if total else 0.0,
                "hits": metrics[k]["hits"],
                "samples": total,
                "playlist_embedding_similarity": _mean_similarity(
                    similarity_metrics, k, "playlist_embedding_similarity"
                ),
                "target_embedding_similarity": _mean_similarity(
                    similarity_metrics, k, "target_embedding_similarity"
                ),
                "playlist_audio_similarity": _mean_similarity(
                    similarity_metrics, k, "playlist_audio_similarity"
                ),
                "random_playlist_embedding_similarity": _mean_similarity(
                    similarity_metrics, k, "random_playlist_embedding_similarity"
                ),
                "random_target_embedding_similarity": _mean_similarity(
                    similarity_metrics, k, "random_target_embedding_similarity"
                ),
                "random_playlist_audio_similarity": _mean_similarity(
                    similarity_metrics, k, "random_playlist_audio_similarity"
                ),
            }
        )

    return rows


def print_report(rows, total, stage2_enabled):
    print(f"Оценено плейлистов: {total}")
    print(f"Stage2 включён: {'да' if stage2_enabled else 'нет'}")
    print(f"Лимит памяти DuckDB: {EVALUATION_MEMORY_LIMIT}")
    print()
    print(",".join(REPORT_FIELDS))

    for row in rows:
        print(",".join(_format_report_value(row[field]) for field in REPORT_FIELDS))


def _format_report_value(value):
    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def save_report(rows, output_path, total, stage2_enabled, args):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        # JSON сохраняет не только таблицу метрик, но и параметры запуска,
        # чтобы в отчёте было понятно, при каких настройках получены числа.
        payload = {
            "samples": total,
            "stage2_enabled": stage2_enabled,
            "candidate_k": args.candidate_k,
            "stage2_k": args.stage2_k,
            "metrics": rows,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if output_path.suffix.lower() != ".csv":
        raise ValueError("Файл отчёта должен иметь расширение .csv или .json.")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=REPORT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    k_values = parse_k_values(args.k_values)

    if not STAGE1_TRACK_VECTORS_NPY.exists():
        print(
            "Предупреждение: кеш векторов Stage1 не найден. "
            "Для ускорения оценки запустите `python scripts/build_stage1_track_vectors.py`.",
            file=sys.stderr,
        )

    # Модели и карты идентификаторов загружаются один раз перед циклом оценки.
    # Это быстрее и экономнее, чем перечитывать веса для каждого плейлиста.
    stage1_assets = load_stage1_assets()
    stage3_assets = load_stage3_assets()

    stage2_assets = None
    stage2_enabled = TRANSFORMER_MODEL_PT.exists()

    if stage2_enabled:
        try:
            stage2_assets = load_stage2_assets()
        except Exception as exc:
            stage2_enabled = False
            print(f"Предупреждение: checkpoint Stage2 найден, но не загрузился: {exc}", file=sys.stderr)

    stage1_metrics = new_metrics(k_values)
    stage2_metrics = new_metrics(k_values)
    final_metrics = new_metrics(k_values)
    stage1_similarity_metrics = new_similarity_metrics(k_values)
    stage2_similarity_metrics = new_similarity_metrics(k_values)
    final_similarity_metrics = new_similarity_metrics(k_values)
    rng = np.random.default_rng(42)
    max_k = max(k_values)
    total = 0

    for playlist_track_ids, target_track_id in stream_eval_playlists(
        db_path=str(PLAYLIST_TRACKS_PARQUET),
        stage1_track_map=stage1_assets["track_map"],
        eval_mod=args.eval_mod,
        eval_remainder=args.eval_remainder,
        max_samples=args.samples,
        duckdb_memory_limit=EVALUATION_MEMORY_LIMIT,
    ):
        # Stage1 формирует первичный список кандидатов. Если собран кеш
        # stage1_track_vectors.npy, поиск выполняется по готовым векторам.
        stage1_candidates = generate_stage1_candidates(
            playlist_ids=playlist_track_ids,
            top_k=args.candidate_k,
            assets=stage1_assets,
        )
        stage1_ids = [item["track_id"] for item in stage1_candidates]
        update_metrics(stage1_metrics, stage1_ids, target_track_id, k_values)
        random_ids = sample_random_track_ids(
            stage1_assets=stage1_assets,
            excluded_track_ids=set(playlist_track_ids) | {target_track_id},
            count=max_k,
            rng=rng,
        )
        update_similarity_metrics(
            metrics=stage1_similarity_metrics,
            ranked_track_ids=stage1_ids,
            random_track_ids=random_ids,
            playlist_track_ids=playlist_track_ids,
            target_track_id=target_track_id,
            stage1_assets=stage1_assets,
            k_values=k_values,
        )

        if stage2_enabled and stage2_assets is not None:
            # Stage2 пересортировывает кандидатов с учётом порядка треков
            # во входном плейлисте.
            stage2_scores = score_sequence_model(
                model=stage2_assets["model"],
                playlist_track_ids=playlist_track_ids,
                candidate_track_ids=stage1_ids,
                track_map=stage2_assets["track_map"],
                pad_idx=stage2_assets["pad_idx"],
                device=stage2_assets["device"],
                max_len=stage2_assets["seq_len"],
            )
            stage2_ids = sorted(
                stage1_ids,
                key=lambda track_id: stage2_scores.get(track_id, 0.0),
                reverse=True,
            )[: args.stage2_k]
        else:
            stage2_ids = stage1_ids[: args.stage2_k]

        update_metrics(stage2_metrics, stage2_ids, target_track_id, k_values)
        update_similarity_metrics(
            metrics=stage2_similarity_metrics,
            ranked_track_ids=stage2_ids,
            random_track_ids=random_ids,
            playlist_track_ids=playlist_track_ids,
            target_track_id=target_track_id,
            stage1_assets=stage1_assets,
            k_values=k_values,
        )

        # Stage3 выполняет финальный reranking и формирует порядок, который
        # считается итоговым качеством всей рекомендательной системы.
        final_ranked = rerank_candidates(
            model=stage3_assets["model"],
            playlist_track_ids=playlist_track_ids,
            candidate_track_ids=stage2_ids,
            filtered_track_map=stage3_assets["track_map"],
            pad_idx=stage3_assets["pad_idx"],
            device=stage3_assets["device"],
        )
        final_ids = [item["track_id"] for item in final_ranked]
        update_metrics(final_metrics, final_ids, target_track_id, k_values)
        update_similarity_metrics(
            metrics=final_similarity_metrics,
            ranked_track_ids=final_ids,
            random_track_ids=random_ids,
            playlist_track_ids=playlist_track_ids,
            target_track_id=target_track_id,
            stage1_assets=stage1_assets,
            k_values=k_values,
        )
        total += 1

    rows = []
    rows.extend(flatten_metrics("stage1", stage1_metrics, stage1_similarity_metrics, total, k_values))
    rows.extend(flatten_metrics("stage2", stage2_metrics, stage2_similarity_metrics, total, k_values))
    rows.extend(flatten_metrics("final_pipeline", final_metrics, final_similarity_metrics, total, k_values))

    return rows, total, stage2_enabled


def main():
    args = parse_args()
    rows, total, stage2_enabled = evaluate(args)
    print_report(rows, total, stage2_enabled)

    if args.output is not None:
        save_report(rows, args.output, total, stage2_enabled, args)
        print(f"\nОтчёт сохранён: {args.output}")


if __name__ == "__main__":
    main()
