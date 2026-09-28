"""Пересчитать три ключевых эксперимента по сохранённым OOF-предсказаниям.

Это проверка исторических результатов, а не повторное обучение. Все три
эксперимента используют те же 8000 запросов и полные знаменатели Recall@50.
"""

import json
from pathlib import Path

import numpy as np

from solution import sha256

ROOT = Path(__file__).resolve().parent


def evaluate():
    report = json.loads((ROOT / "report.json").read_text(encoding="utf-8"))
    path = ROOT / "validation.npz"
    if sha256(path) != report["validation"]["data_sha256"]:
        raise ValueError("Изменился сохранённый набор для валидации")
    with np.load(ROOT / "bundle/indices.npz", allow_pickle=False) as indices:
        ids = indices["item_ids"]
    with np.load(path, allow_pickle=False) as data:
        query_ids, folds = data["query_ids"], data["folds"]
        if len(query_ids) != 8000 or len(np.unique(query_ids)) != 8000:
            raise ValueError("Изменился набор из 8000 запросов")
        if np.unique(data["query_keys"]).size != 8000:
            raise ValueError("Повторяются нормализованные тексты запросов")
        if np.intersect1d(data["query_keys"], data["history_keys"]).size:
            raise ValueError("Проверочные запросы присутствуют в истории обучения")
        label_query, label_item = data["label_query"], data["label_item"]
        if len(label_query) != 8493 or not np.array_equal(np.unique(label_query), np.arange(8000)):
            raise ValueError("Изменилась локальная разметка")
        truth = [set(ids[label_item[label_query == n]]) for n in range(8000)]
        results = []
        for experiment in report["experiments"]:
            predictions = data[experiment["key"]]
            if predictions.shape != (8000, 50):
                raise ValueError("Неверная форма OOF-предсказаний")
            values = []
            for n, indices in enumerate(predictions):
                found = set(ids[indices])
                if len(found) != 50:
                    raise ValueError("Повторяются кандидаты")
                values.append(len(found & truth[n]) / len(truth[n]))
            values = np.asarray(values)
            recall = float(values.mean())
            if not np.isclose(recall, experiment["recall50"], atol=1e-12, rtol=0):
                raise ValueError(f"Не воспроизвелась метрика: {experiment['key']}")
            results.append(
                {
                    "experiment": experiment["key"],
                    "recall50": recall,
                    "folds": [float(values[folds == k].mean()) for k in range(4)],
                }
            )
    return {
        "queries": 8000,
        "labels": 8493,
        "history_query_overlap": 0,
        "selection_used_oof": True,
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
