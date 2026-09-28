"""Повторить отправленный answer.csv: python predict.py (CPU, без сети)."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from solution import read_manifest, verify_bundle, load_data
from solution import Rankers
from solution import combine, duplicate_groups, select_answers
from solution import validate_submission

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the submitted CSV from saved models (CPU)."
    )
    parser.add_argument("--output", type=Path, default=ROOT / "answer.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    bundle = ROOT / "bundle"
    manifest = read_manifest(bundle)
    verify_bundle(bundle, manifest)
    queries, items = load_data(args.data_dir, bundle, manifest)
    groups = duplicate_groups(items)
    models = Rankers(bundle, manifest)
    answers = []
    for number, name in enumerate(manifest["shards"], 1):
        with np.load(bundle / name, allow_pickle=False) as data:
            scores = models.predict(data)
            combined = combine(scores, data["query_index"], data["item_index"], groups)
            answers.extend(
                select_answers(combined, data["query_index"], data["item_index"], queries, items)
            )
        print(f"Processed shards: {number}/{len(manifest['shards'])}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    pd.DataFrame(answers).to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    report = validate_submission(temporary, queries, items)
    if report["sha256"] != manifest["expected_answer_sha256"]:
        raise ValueError("CSV не совпадает побайтно с отправленным. Старый результат сохранён.")
    os.replace(temporary, args.output)
    (args.output.parent / "prediction_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
