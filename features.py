"""Сборка сохранённых признаков v4 для переносимого снимка.

Вход — промежуточные результаты подготовки текстов, эмбеддингов и статистик.
Экспорт не обучается на benchmark и не использует его ответы.
"""

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_features(root: Path):
    candidates = np.load(root / "candidates_tuned.npz")
    x = candidates["x"]
    qi, ii = candidates["qi"], candidates["ii"]
    queries = pd.read_parquet(root / "queries.parquet")
    items = pd.read_parquet(root / "corpus.parquet")
    names = json.loads((root / "feature_names_tuned.json").read_text())

    # Вероятность категории, предсказанная только по обучающей истории.
    category = np.load(root / "category.npz")
    lookup = {int(value): j for j, value in enumerate(category["classes"])}
    column = np.array([lookup.get(int(value), -1) for value in items.item_microcat_id])
    probability = category["p"][qi, np.maximum(column[ii], 0)]
    probability[column[ii] < 0] = 0
    maximum = category["p"].max(axis=1)[qi]
    x = np.column_stack([x, probability, probability / np.maximum(maximum, 1e-8)])
    names += ["category_probability", "category_relative_probability"]

    remote = np.load(root / "remote_probability.npy")
    entropy = -(category["p"] * np.log(np.maximum(category["p"], 1e-10))).sum(axis=1)
    x = np.column_stack([x, remote[qi], maximum, entropy[qi]])
    names += ["remote_probability", "category_confidence", "category_entropy"]
    x = np.column_stack([x, np.load(root / "filter_features_tuned.npy")])
    names += ["filter_word_coverage", "required_rating", "rating_filter_match"]

    # Лексика, география, относительные характеристики и ранги каналов поиска.
    x = np.column_stack([x, np.load(root / "v3_features.npy")])
    names += json.loads((root / "v3_feature_names.json").read_text())

    # Прототипы — средние эмбеддинги объявлений из похожих запросов истории.
    prototype_root = root / "goal/prototypes"
    x = np.column_stack([x, np.load(prototype_root / "original_features.npy")])
    names += json.loads((prototype_root / "audit.json").read_text())["features"]
    confidence = np.load(root / "goal/neighbors/probabilities.npz")["similarity"][: len(queries)]
    prototype = np.load(prototype_root / "query_prototypes.npz")["mean8"][: len(queries)].astype(
        np.float32
    )
    query_vectors = np.load(root / "queries_e5.npy").astype(np.float32)
    agreement = (prototype * query_vectors).sum(axis=1)
    x = np.column_stack([x, confidence[qi], agreement[qi]])
    names += ["prototype_neighbor_confidence", "prototype_query_agreement"]
    if x.dtype != np.float32 or x.shape[1] != 79 or not np.isfinite(x).all():
        raise ValueError("Неверная матрица исходных признаков")
    return x, qi, ii, names


def reference_ranks(x, qi, reference_root: Path):
    """Ранг нового кандидата относительно того пула, на котором учили модель."""
    pool = np.load(reference_root / "candidates_tuned.npz")
    reference = pool["x"]
    rq = pool["qi"]
    starts = np.r_[0, np.flatnonzero(np.diff(qi)) + 1]
    ends = np.r_[starts[1:], len(qi)]
    rstarts = np.r_[0, np.flatnonzero(np.diff(rq)) + 1]
    rends = np.r_[rstarts[1:], len(rq)]
    if not np.array_equal(qi[starts], rq[rstarts]):
        raise ValueError("В пулах сравнения разные запросы")
    for target, source in enumerate([0, 1, 2, 3, 28, 30, 31], 64):
        for a, b, ra, rb in zip(starts, ends, rstarts, rends):
            rank = (
                np.searchsorted(np.sort(-reference[ra:rb, source]), -x[a:b, source], side="left")
                + 1
            )
            x[a:b, target] = np.log1p(rank)
    del pool, reference
    gc.collect()
