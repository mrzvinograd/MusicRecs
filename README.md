# MusicRecs

MusicRecs - рекомендательная система для музыкальных треков. Проект построен как многоэтапный ML-пайплайн: подготовка данных, обучение эмбеддингов, retrieval-модель, sequence-модель, финальный reranking и Streamlit-интерфейс для получения рекомендаций.

## Быстрый запуск приложения

В корне репозитория есть файл `main.py`, который запускает приложение для работы с моделью:

```bash
python main.py
```

Команда запускает Streamlit-интерфейс без необходимости вручную писать `streamlit run app.py`.

Если Python не видит зависимости, установите их:

```bash
pip install -r requirements.txt
```

## Ускорение рекомендаций

Чтобы не ждать 10-15 минут при выдаче рекомендаций, stage1 поддерживает кеш уже закодированных векторов треков. Его нужно собрать один раз после обучения `two_tower.pt`:

```bash
python scripts/build_stage1_track_vectors.py
```

После этого файл `data/processed/stage1_track_vectors.npy` будет использоваться автоматически, а во время рекомендации stage1 будет выполнять быстрый поиск по готовым векторам вместо повторного прогона всего каталога через модель.

## Математическое описание моделей

В проекте используются три основные модели, которые последовательно формируют итоговую рекомендацию.

### Stage1: Two-Tower Retrieval

На первом этапе плейлист пользователя и треки переводятся в общее векторное пространство. Пусть входной плейлист задан последовательностью треков:

```text
X = (x_1, x_2, ..., x_m)
```

Для плейлиста строится вектор:

```text
p = normalize(W_2 * ReLU(W_1 * h_m + b_1) + b_2),
```

где `h_m` - последнее скрытое состояние LSTM после обработки последовательности треков плейлиста. Для кандидата `i` с признаковым вектором `z_i` строится track-вектор:

```text
t_i = normalize(W_4 * ReLU(W_3 * z_i + b_3) + b_4).
```

Оценка релевантности трека вычисляется скалярным произведением:

```text
s_i = p^T * t_i.
```

В рекомендации передаются `K` треков с наибольшими значениями `s_i`, исключая треки, которые уже есть во входном плейлисте.

### Stage2: Transformer Sequence Model

На втором этапе учитывается порядок треков в плейлисте. Для каждого трека входной последовательности суммируется embedding трека и позиционный embedding:

```text
u_j = E(x_j) + P_j.
```

После обработки Transformer Encoder получается скрытое представление последовательности:

```text
H = Transformer(u_1, u_2, ..., u_m).
```

Вектор плейлиста берётся по последней валидной позиции и нормализуется:

```text
q = normalize(H_m).
```

Для кандидата `i` используется нормализованный embedding:

```text
v_i = normalize(E(i)).
```

Оценка кандидата вычисляется как:

```text
s_i = q^T * v_i.
```

Если модель используется как классификатор следующего трека по всему словарю, вероятность трека можно записать через softmax:

```text
P(i | X) = exp(q^T * v_i) / sum_j exp(q^T * v_j).
```

### Stage3: Ranking Model

На третьем этапе выполняется финальная сортировка кандидатов. Последовательность плейлиста кодируется LSTM:

```text
O, h_m = LSTM(E(x_1), E(x_2), ..., E(x_m)).
```

Дополнительно считается среднее по валидным выходам LSTM:

```text
o_mean = (1 / m) * sum_j O_j.
```

Итоговое представление плейлиста:

```text
p = phi_p([h_m; o_mean]).
```

Для кандидата `i` строится представление трека:

```text
r_i = phi_t(E(i)).
```

Финальный скоринг выполняется MLP по объединению признаков плейлиста, трека и их поэлементного произведения:

```text
y_i = f([p; r_i; p ⊙ r_i]).
```

Чем больше значение `y_i`, тем выше трек располагается в итоговом списке рекомендаций.

## Структура проекта

- `main.py` - основной запуск приложения MusicRecs.
- `app.py` - Streamlit-интерфейс рекомендательной системы.
- `config.py` - централизованные пути к данным, моделям и артефактам.
- `data/raw/` - локальные исходные данные. Основные parquet-файлы Spotify сейчас берутся из `D:\ZZZFORALL\dbmusic`, если локальные файлы отсутствуют.
- `data/processed/` - подготовленные parquet-файлы, индексы, чекпоинты моделей и кеши для инференса.
- `data/embeddings/` - `track2vec`-модель, эмбеддинги и карты идентификаторов.
- `stage0/` - обучение `track2vec` и экспорт эмбеддингов.
- `stage1/` - two-tower retrieval: обучение, генерация кандидатов и кеш track-векторов.
- `stage2/` - transformer-модель для оценки последовательности треков.
- `stage3/` - финальная ranking-модель, end-to-end рекомендации и оценка качества.
- `scripts/` - скрипты подготовки данных, индексов и кешей.
- `utils/` - общие функции для метаданных и аудио-признаков.

## Основные команды

Подготовка индекса Spotify ID для интерфейса:

```bash
python scripts/build_track_search_index.py
```

Обучение `track2vec`:

```bash
python stage0/train_track2vec.py
python stage0/export_embeddings.py
python stage0/extract_mapping.py
python stage0/validate_artifacts.py
```

Подготовка данных:

```bash
python scripts/build_track_frequency.py
python scripts/build_filtered_sessions.py
python scripts/build_track_map.py
python scripts/build_filtered_audio.py
python scripts/prepare_feature_indices.py
```

Обучение моделей:

```bash
python stage1/train_two_tower.py
python stage2/train_transformer.py
python stage3/train_ranking.py
```

Сборка кеша для быстрого stage1-инференса:

```bash
python scripts/build_stage1_track_vectors.py
```

Запуск рекомендаций из консоли:

```bash
python stage3/recommend_system.py --playlist "10000,1178,2779" --top-k 20
```

Запуск UI:

```bash
python main.py
```

## Параметры интерфейса

- `Кандидатов Stage1` - сколько первичных кандидатов retrieval-модель выбирает из полного каталога.
- `Оставить после Stage2` - сколько кандидатов после sequence scoring передаётся в финальный ranking.
- `Показать top-k` - сколько финальных рекомендаций отображается пользователю.

## Полный порядок переобучения

1. Подготовить частотную базу топ-треков:
   `python scripts/build_track_frequency.py`
2. Зафиксировать отфильтрованное пространство плейлистов:
   `python scripts/build_filtered_sessions.py`
3. Собрать карту track id:
   `python scripts/build_track_map.py`
4. Экспортировать аудио-признаки:
   `python scripts/build_filtered_audio.py`
5. Подготовить нормализованные аудио-массивы:
   `python scripts/prepare_feature_indices.py`
6. Обучить `track2vec`:
   `python stage0/train_track2vec.py`
7. Экспортировать эмбеддинги:
   `python stage0/export_embeddings.py`
8. Экспортировать карту stage0:
   `python stage0/extract_mapping.py`
9. Проверить артефакты stage0:
   `python stage0/validate_artifacts.py`
10. Собрать индекс поиска Spotify ID:
    `python scripts/build_track_search_index.py`
11. Обучить stage1 retrieval:
    `python stage1/train_two_tower.py`
12. Обучить stage2 transformer:
    `python stage2/train_transformer.py`
13. Обучить stage3 reranker:
    `python stage3/train_ranking.py`
14. Собрать кеш track-векторов для быстрого инференса:
    `python scripts/build_stage1_track_vectors.py`
15. Запустить end-to-end оценку:
    `python stage3/eval_system.py --samples 50 --candidate-k 300 --stage2-k 100`
16. Проверить рекомендации:
    `python stage3/recommend_system.py --playlist "10000,1178,2779" --top-k 20`
17. Запустить приложение:
    `python main.py`

## English Summary

MusicRecs is a multi-stage music recommendation system with a Streamlit UI. The main application entry point is:

```bash
python main.py
```

For faster recommendations, build the stage1 vector cache once:

```bash
python scripts/build_stage1_track_vectors.py
```
