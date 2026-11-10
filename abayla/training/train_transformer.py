"""Обучение XLM-RoBERTa на тех же разбиениях, что и baseline (сначала запустите abayla.training.train).

    python -m abayla.training.train_transformer --epochs 4
На GPU (Colab): см. notebooks/train_xlmr_colab.ipynb, можно добавить --no-freeze-embeddings.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from abayla.config import ROOT
from abayla.data.prepare import load_split
from abayla.models.transformer import TransformerClassifier
from abayla.training.env_info import hardware_info, package_versions
from abayla.training.thresholds import select_thresholds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(ROOT / "artifacts" / "splits"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "models" / "xlmr"))
    ap.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-freeze-embeddings", action="store_true")
    args = ap.parse_args(argv)

    split_dir, out = Path(args.splits), Path(args.out)
    x_tr, y_tr, _ = load_split(split_dir, "train")
    x_va, y_va, _ = load_split(split_dir, "val")
    info = json.loads((split_dir / "split_info.json").read_text(encoding="utf-8"))
    m = TransformerClassifier(model_name=args.model_name, max_len=args.max_len, epochs=args.epochs, lr=args.lr,
                              batch_size=args.batch_size, freeze_embeddings=not args.no_freeze_embeddings,
                              seed=args.seed)
    t0 = time.perf_counter()
    m.fit(x_tr, y_tr, x_va, y_va, checkpoint_dir=out)
    m.meta.train_time_s = round(time.perf_counter() - t0, 1)
    m.meta.thresholds = select_thresholds(y_va, m.predict_scores(x_va))
    m.meta.data_version = f"{Path(info['dataset']).stem}@{info['data_hash']}"
    m.meta.demo_data = "demo" in info["dataset"]
    m.meta.hardware = hardware_info()
    m.meta.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    m.meta.extra.update({"packages": package_versions(), "n_train": len(x_tr), "n_val": len(x_va)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(m.meta.to_json(), encoding="utf-8")  # веса уже сохранены лучшей эпохой
    print("thresholds:", m.meta.thresholds, "train_time_s:", m.meta.train_time_s)


if __name__ == "__main__":
    main()
