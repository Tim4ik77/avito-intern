"""Модели, точная формула v4 и проверка результата. Только локальный CPU."""

import hashlib
import json
import re
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(bundle: Path) -> dict:
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


def verify_bundle(bundle: Path, manifest: dict) -> None:
    """Не даём случайно смешать признаки и веса разных версий."""
    base = bundle.resolve()
    for name, expected in manifest["files"].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError(f"Нет файла снимка: {name}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Изменился файл снимка: {name}")


def load_data(data_dir: Path, bundle: Path, manifest: dict):
    """Берём метаданные из исходных Parquet, порядок строк — из снимка v4."""
    for name, expected in manifest["input_files"].items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Put {name} in {data_dir}. See README.md.")
        if sha256(path) != expected:
            raise ValueError(f"Input differs from the competition dataset: {name}")
    columns = ["item_id", "item_title_raw", "item_location_id"]
    benchmark = pd.read_parquet(data_dir / "benchmark_items.parquet", columns=columns)
    history = pd.read_parquet(data_dir / "train.parquet", columns=columns)
    queries = pd.read_parquet(data_dir / "benchmark_queries.parquet", columns=["query_id"])
    with np.load(bundle / "indices.npz", allow_pickle=False) as indices:
        items = (
            pd.concat([benchmark, history])
            .drop_duplicates("item_id")
            .set_index("item_id")
            .loc[indices["item_ids"]]
            .reset_index()
        )
        queries = queries.set_index("query_id").loc[indices["query_ids"]].reset_index()
    items["allowed"] = items.item_id.isin(benchmark.item_id)
    return queries, items


class Rankers:
    def __init__(self, bundle: Path, manifest: dict):
        self.models = {}
        self.specs = manifest["models"]
        names = manifest["feature_names"]
        for name, spec in self.specs.items():
            path = bundle / spec["file"]
            if spec["library"] == "lightgbm":
                model = lgb.Booster(model_file=str(path))
                actual_names = model.feature_name()
            else:
                model = CatBoostClassifier()
                model.load_model(str(path))
                actual_names = model.feature_names_
            if actual_names != names[: spec["features"]]:
                raise ValueError(f"Неверный порядок признаков модели {name}")
            self.models[name] = model

    def predict(self, shard: dict) -> dict[str, np.ndarray]:
        common = shard["features"]
        # У последней модели ранги считались по расширенному набору кандидатов.
        # Остальные три обучались с рангами исходного набора. Это важно сохранить.
        adapted = np.column_stack([common, shard["adapted_features"]])
        adapted[:, 64:71] = shard["expanded_ranks"]
        scores = {}
        for name, model in self.models.items():
            spec = self.specs[name]
            features = adapted if name == "adapted" else common[:, : spec["features"]]
            if spec["library"] == "lightgbm":
                values = model.predict(features, num_iteration=spec["trees"], num_threads=4)
            else:
                values = model.predict(
                    features,
                    prediction_type="RawFormulaVal",
                    ntree_end=spec["trees"],
                    thread_count=4,
                )
            # Исходная версия сохраняла баллы компонентов именно в float32.
            scores[name] = values.astype(np.float32)
            if not np.isfinite(scores[name]).all():
                raise ValueError(f"Некорректный балл модели {name}")
        return scores


def query_slices(query_indices: np.ndarray):
    if len(query_indices) == 0 or np.any(np.diff(query_indices) < 0):
        raise ValueError("Кандидаты должны быть непустыми и отсортированными по запросу")
    starts = np.r_[0, np.flatnonzero(np.diff(query_indices)) + 1]
    return zip(starts, np.r_[starts[1:], len(query_indices)])


def duplicate_groups(items: pd.DataFrame) -> np.ndarray:
    # Сохраняем исторический ASCII-ключ v4. В исходном pandas/Arrow \w был
    # ASCII: кириллица удалялась. Исправлять это правило при упаковке нельзя —
    # изменится отправленный ответ. Явный класс даёт одинаковый результат
    # независимо от выбранного строкового движка pandas.
    title = (
        items.item_title_raw.fillna("")
        .str.lower()
        .str.replace("ё", "е")
        .str.replace(r"[^a-z0-9_]+", " ", regex=True)
        .str.strip()
    )
    return pd.factorize(title + "|" + items.item_location_id.astype(str))[0]


def standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(values.std(), 1e-6)


def combine(scores: dict, qi: np.ndarray, ii: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.empty(len(qi), np.float32)
    for a, b in query_slices(qi):
        ranks = [
            rankdata(-scores[name][a:b], method="min").astype(np.float32)
            for name in ["lgb", "cat", "prototype"]
        ]
        fusion = sum(weight * (-np.log(rank)) for weight, rank in zip([0.375, 0.125, 0.5], ranks))
        base = standardize(fusion)
        penalty = np.empty(b - a, np.float32)
        seen = {}
        # Порядок обхода и dtype оставлены как в v4: они влияют на равенства.
        for j in np.argsort(-fusion):
            key = int(groups[ii[a + j]])
            count = seen.get(key, 0)
            penalty[j] = np.log1p(count)
            seen[key] = count + 1
        base = base - 0.025 * penalty
        result[a:b] = 0.25 * standardize(base) + 0.75 * standardize(scores["adapted"][a:b])
    return result


def select_answers(score, qi, ii, queries, items):
    ids = items.item_id.to_numpy()
    legal = items.allowed.to_numpy(dtype=bool)
    answers = []
    for a, b in query_slices(qi):
        # Дополнительные локальные документы участвуют в нормализации баллов,
        # но не могут попасть в ответ benchmark.
        eligible = np.flatnonzero(legal[ii[a:b]]) + a
        order = np.lexsort((ids[ii[eligible]], -score[eligible]))
        chosen = eligible[order[:50]]
        if len(chosen) != 50:
            raise ValueError("Для запроса найдено меньше 50 разрешённых объявлений")
        answers.append(
            {
                "query_id": queries.iloc[int(qi[a])].query_id,
                "answer": " ".join(ids[ii[chosen]]),
            }
        )
    return answers


def validate_submission(path: Path, queries: pd.DataFrame, items: pd.DataFrame) -> dict:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != ["query_id", "answer"]:
        raise ValueError("Ожидаются колонки query_id,answer")
    if not frame.query_id.is_unique or set(frame.query_id) != set(queries.query_id):
        raise ValueError("Пропущены, добавлены или повторяются запросы")
    allowed = set(items.loc[items.allowed, "item_id"])
    for row in frame.itertuples(index=False):
        ids = row.answer.split()
        if len(row.query_id) != 16 or len(ids) != 50 or len(set(ids)) != 50:
            raise ValueError(f"Неверный формат ответа: {row.query_id}")
        if not all(re.fullmatch(r"[0-9a-f]{16}", item) for item in ids) or not set(ids) <= allowed:
            raise ValueError(f"Неизвестное объявление: {row.query_id}")
    return {
        "rows": len(frame),
        "candidates_per_query": 50,
        "valid": True,
        "sha256": sha256(path),
    }
