"""Обучение baseline-моделей: правила, TF-IDF(word)+LR, TF-IDF(word+char)+LR.

    python -m abayla.training.train --data data/demo/abayla_demo_v1.csv

1) проверка датасета; исключение disputed/pending;
2) разбиение по template_group 70/15/15 (seed фиксирован);
3) обучение только на train (TF-IDF тоже);
4) подбор порогов только на validation;
5) сохранение модели + meta.json (параметры, seed, версия данных, время, оборудование).
Test здесь не используется.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from abayla.config import ROOT
from abayla.data.prepare import load_split, prepare_splits
from abayla.models.base import ScamClassifier
from abayla.models.rules import RuleClassifier
from abayla.models.tfidf import TfidfLogReg
from abayla.training.env_info import hardware_info, package_versions
from abayla.training.thresholds import select_thresholds

MODEL_FACTORIES = {
    "rules": lambda seed: RuleClassifier(),
    "tfidf_word": lambda seed: TfidfLogReg(analyzer="word", seed=seed),
    "tfidf_wordchar": lambda seed: TfidfLogReg(analyzer="wordchar", seed=seed),
}


def data_version_of(dataset_path: Path, split_info: dict) -> tuple[str, bool]:
    df = pd.read_csv(dataset_path, dtype=str, keep_default_na=False)
    versions = "+".join(sorted(df["source_id"].unique())[:3])
    demo = bool((df["source_type"] == "synthetic_demo").all())
    return f"{versions}@{split_info['data_hash']}", demo


def train_one(name: str, split_dir: Path, out_dir: Path, seed: int, data_version: str, demo: bool) -> ScamClassifier:
    x_tr, y_tr, _ = load_split(split_dir, "train")
    x_va, y_va, _ = load_split(split_dir, "val")
    np.random.seed(seed)
    model = MODEL_FACTORIES[name](seed)
    t0 = time.perf_counter()
    model.fit(x_tr, y_tr)
    train_time = time.perf_counter() - t0
    s_va = model.predict_scores(x_va)
    model.meta.name = name
    model.meta.thresholds = select_thresholds(y_va, s_va)
    model.meta.data_version = data_version
    model.meta.demo_data = demo
    model.meta.seed = seed
    model.meta.train_time_s = round(train_time, 3)
    model.meta.hardware = hardware_info()
    model.meta.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model.meta.extra = {"packages": package_versions(), "n_train": len(x_tr), "n_val": len(x_va)}
    model.save(out_dir / name)
    return model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "demo" / "abayla_demo_v1.csv"))
    ap.add_argument("--splits", default=str(ROOT / "artifacts" / "splits"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "models"))
    ap.add_argument("--models", default="rules,tfidf_word,tfidf_wordchar")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", action="store_true", help="аугментация только train после разбиения")
    args = ap.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists() and "abayla_demo_v1" in data_path.name:
        from abayla.data.demo_dataset import write_demo_csv

        write_demo_csv(data_path)
    split_dir = Path(args.splits)
    info = prepare_splits(data_path, split_dir, seed=args.seed, augment=args.augment)
    print("split:", json.dumps({k: info[k] for k in ("train", "val", "test")}, ensure_ascii=False))
    data_version, demo = data_version_of(data_path, info)
    for name in args.models.split(","):
        m = train_one(name, split_dir, Path(args.out), args.seed, data_version, demo)
        th = m.meta.thresholds
        print(f"[{name}] train {m.meta.train_time_s}s | thresholds binary={th.binary} low={th.low} "
              f"high={th.high} preliminary={th.preliminary}")
    if demo:
        print("ВНИМАНИЕ: модели обучены на демонстрационных синтетических данных. Метрики не отражают реальное качество.")


if __name__ == "__main__":
    main()
