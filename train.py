"""Ключевой код обучения: дообучение E5 и четыре финальных ранжировщика.

Не запускается при проверке отправки. Нужен --workspace с подготовленными
artifacts/ и локальными models/. Для E5 и CatBoost требуется GPU.
"""

import os

from pathlib import Path
import argparse
import random
import time
import numpy as np
import pandas as pd


def finetune(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(6)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    root = args.workspace.resolve()
    out = root / "artifacts"
    h = pd.read_parquet(out / "history.parquet").fillna("")
    # Cap head-query frequency, and remove exact pair duplicates.
    h = (
        h[h.norm_query != ""]
        .drop_duplicates(["norm_query", "item_id"])
        .sample(frac=1, random_state=42)
    )
    h = h.groupby("norm_query", sort=False).head(3).reset_index(drop=True)
    model = SentenceTransformer(str(root / "models" / args.model), device="cuda")
    model.max_seq_length = 160
    local = pd.read_parquet(out / "queries.parquet")
    assert not set(h.norm_query) & set(local.loc[local.split != "benchmark", "norm_query"])
    if args.freeze_embeddings:
        model[0].auto_model.embeddings.requires_grad_(False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.bf16)
    query = ("query: " + h.search_query).tolist()
    passage = (
        "passage: " + h.item_title_raw + ". " + h.item_description_raw.str.slice(0, 1500)
    ).tolist()
    qkey = h.norm_query.to_numpy()
    ikey = h.item_id.to_numpy()
    rng = np.random.default_rng(42)
    order = rng.permutation(len(h))
    cursor = 0
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        if cursor + args.batch > len(h):
            order = rng.permutation(len(h))
            cursor = 0
        ix = order[cursor : cursor + args.batch]
        cursor += args.batch
        qt = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in model.tokenize([query[i] for i in ix]).items()
        }
        pt = {
            k: (v.cuda() if torch.is_tensor(v) else v)
            for k, v in model.tokenize([passage[i] for i in ix]).items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16 if args.bf16 else torch.float16):
            a = torch.nn.functional.normalize(model(qt)["sentence_embedding"], dim=1)
            b = torch.nn.functional.normalize(model(pt)["sentence_embedding"], dim=1)
            logits = (a @ b.T).float() / 0.035
            # Other positives for the same query/item must not become negatives.
            bad = (qkey[ix, None] == qkey[ix][None, :]) | (ikey[ix, None] == ikey[ix][None, :])
            np.fill_diagonal(bad, False)
            logits = logits.masked_fill(torch.tensor(bad, device="cuda"), -1e4)
            target = torch.arange(len(ix), device="cuda")
            loss = (
                torch.nn.functional.cross_entropy(logits, target)
                + torch.nn.functional.cross_entropy(logits.T, target)
            ) * 0.5
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite contrastive training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = 1.5e-5 * min(1, (step + 1) / 100) * max(0.1, 1 - step / args.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            print(
                step + 1,
                "loss",
                round(np.mean(losses[-50:]), 4),
                "seconds",
                round(time.time() - t0),
                flush=True,
            )
    model.save(str(root / "models" / args.output))
    print("Saved adapted model", flush=True)


def fit_ranker(args):
    """Повтор финального обучения после выбора параметров на четырёх фолдах."""
    import json
    import lightgbm as lgb
    from catboost import CatBoostClassifier, Pool
    from features import load_features

    if args.output.exists():
        raise FileExistsError("Выберите новый путь: готовые веса не перезаписываются")
    artifacts = args.workspace.resolve() / "artifacts"
    root = artifacts / "goal/expanded" if args.kind == "adapted" else artifacts
    x, qi, _, names = load_features(root)
    y = np.load(root / "candidates_tuned.npz")["y"]
    folds = np.load(artifacts / "cv/query_fold.npy")
    train = folds[qi] >= 0
    assert np.unique(qi[train]).size == 8000
    queries = pd.read_parquet(artifacts / "queries.parquet")
    counts = pd.read_parquet(artifacts / "labels.parquet").groupby("query_id").size()
    report = json.loads((Path(__file__).parent / "report.json").read_text(encoding="utf-8"))

    if args.kind in {"lgb", "cat"}:
        x, names = x[:, :71], names[:71]
        params = report["training"][args.kind]["parameters"]
        if args.kind == "cat":
            params["gpu_ram_part"] = 0.4
    else:
        params = report["training"][args.kind]["parameters"]

    if args.kind == "adapted":
        extra = np.load(root / "goal/adapted_base/features.npy")
        x = np.column_stack([x, extra])
        names += json.loads((root / "goal/adapted_base/names.json").read_text())
        teacher = lgb.Booster(model_file=str(artifacts / "v3/features_final.txt"))
        indices = np.flatnonzero(train)
        score = teacher.predict(x[indices, :71], num_threads=4)
        groups = qi[indices]
        starts = np.r_[0, np.flatnonzero(np.diff(groups)) + 1]
        ends = np.r_[starts[1:], len(indices)]
        keep = np.zeros(len(qi), bool)
        rng = np.random.default_rng(74)
        for a, b in zip(starts, ends):
            rows = indices[a:b]
            positives = rows[y[rows] > 0]
            negatives = np.flatnonzero(y[rows] == 0)
            count = min(200, len(negatives))
            hard = negatives[np.argpartition(score[a:b][negatives], -count)[-count:]]
            remaining = np.setdiff1d(negatives, hard)
            random_negatives = rng.choice(remaining, min(64, len(remaining)), replace=False)
            keep[positives] = True
            keep[rows[hard]] = True
            keep[rows[random_negatives]] = True
        assert not np.any(keep & ~train) and y[keep].sum() == y[train].sum()
        train = keep

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind in {"cat", "prototype"}:
        weights = np.array(
            [1 / counts[queries.iloc[int(n)].query_id] for n in qi[train]], np.float32
        )
        dataset = Pool(x[train], y[train], weight=weights, feature_names=names)
        model = CatBoostClassifier(**params)
        model.fit(dataset)
        model.save_model(str(args.output))
    else:
        dataset = lgb.Dataset(
            x[train],
            label=y[train],
            group=np.unique(qi[train], return_counts=True)[1],
            feature_name=names,
        )
        model = lgb.train(params, dataset, num_boost_round=600 if args.kind == "adapted" else 500)
        model.save_model(str(args.output))
    print("Saved", args.output, "training rows", int(train.sum()))


def main():
    parser = argparse.ArgumentParser(
        description="Optional training on prepared research artifacts."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    encoder = commands.add_parser("encoder", help="Fine-tune E5 using training history")
    encoder.add_argument("--workspace", type=Path, required=True)
    encoder.add_argument("--steps", type=int, default=1200)
    encoder.add_argument("--batch", type=int, default=96)
    encoder.add_argument("--model", default="e5-small")
    encoder.add_argument("--output", default="e5-tuned")
    encoder.add_argument("--freeze-embeddings", action="store_true")
    encoder.add_argument("--bf16", action="store_true")
    ranker = commands.add_parser("ranker", help="Train a ranker on prepared features")
    ranker.add_argument("--workspace", type=Path, required=True)
    ranker.add_argument("--kind", choices=["lgb", "cat", "prototype", "adapted"], required=True)
    ranker.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "encoder":
        finetune(args)
    else:
        fit_ranker(args)


if __name__ == "__main__":
    main()
