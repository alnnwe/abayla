"""Модель №4: дообученный XLM-RoBERTa (классификация последовательности, 2 класса).

Реализован тем же интерфейсом ScamClassifier, поэтому бот переключается на него
сменой SAQBOL_MODEL_DIR=artifacts/models/xlmr без изменения кода.

Ограничение среды разработки: 1 CPU и 3.9 ГБ RAM. Поэтому по умолчанию заморожены эмбеддинги
(192 млн из 278 млн параметров): иначе состояния Adam не помещаются в память.
На GPU можно обучать полностью: --no-freeze-embeddings (см. notebooks/train_xlmr_colab.ipynb).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from saqbol.models.base import ModelMeta, ScamClassifier


class TransformerClassifier(ScamClassifier):
    model_type = "transformer"

    def __init__(self, model_name: str = "FacebookAI/xlm-roberta-base", max_len: int = 128, epochs: int = 4,
                 lr: float = 3e-5, batch_size: int = 8, weight_decay: float = 0.01, warmup_ratio: float = 0.1,
                 freeze_embeddings: bool = True, seed: int = 42, meta: ModelMeta | None = None):
        params = dict(model_name=model_name, max_len=max_len, epochs=epochs, lr=lr, batch_size=batch_size,
                      weight_decay=weight_decay, warmup_ratio=warmup_ratio, freeze_embeddings=freeze_embeddings,
                      selection="лучшая эпоха по PR-AUC на validation")
        super().__init__(meta or ModelMeta(model_type="transformer", name="xlmr", seed=seed, params=params))
        self.model = None
        self.tokenizer = None
        self.history: list[dict] = []

    def _device(self):
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        p = self.meta.params
        self.tokenizer = AutoTokenizer.from_pretrained(p["model_name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(p["model_name"], num_labels=2)
        if p["freeze_embeddings"]:
            for prm in self.model.base_model.embeddings.parameters():
                prm.requires_grad = False
        self.model.to(self._device())

    def _batches(self, texts, bs):
        for i in range(0, len(texts), bs):
            enc = self.tokenizer(list(texts[i:i + bs]), truncation=True, max_length=self.meta.params["max_len"],
                                 padding=True, return_tensors="pt")
            yield i, {k: v.to(self._device()) for k, v in enc.items()}

    def fit(self, texts, y, val_texts=None, val_y=None, checkpoint_dir: Path | None = None):
        import torch
        from sklearn.metrics import average_precision_score

        p = self.meta.params
        torch.manual_seed(self.meta.seed)
        np.random.seed(self.meta.seed)
        self._init()
        trainable = [prm for prm in self.model.parameters() if prm.requires_grad]
        self.meta.extra["trainable_params"] = int(sum(prm.numel() for prm in trainable))
        opt = torch.optim.AdamW(trainable, lr=p["lr"], weight_decay=p["weight_decay"])
        steps = math.ceil(len(texts) / p["batch_size"]) * p["epochs"]
        warm = max(1, int(steps * p["warmup_ratio"]))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
        texts, y = list(texts), np.asarray(y)
        rng = np.random.default_rng(self.meta.seed)
        best = -1.0
        for epoch in range(1, p["epochs"] + 1):
            self.model.train()
            order = rng.permutation(len(texts))
            t0, losses = time.perf_counter(), []
            for i in range(0, len(order), p["batch_size"]):
                idx = order[i:i + p["batch_size"]]
                enc = self.tokenizer([texts[j] for j in idx], truncation=True, max_length=p["max_len"],
                                     padding=True, return_tensors="pt").to(self._device())
                labels = torch.tensor(y[idx], dtype=torch.long, device=self._device())
                out = self.model(**enc, labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                losses.append(out.loss.item())
            rec = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                   "epoch_time_s": round(time.perf_counter() - t0, 1)}
            if val_texts is not None:
                s = self.predict_scores(val_texts)
                rec["val_pr_auc"] = round(float(average_precision_score(val_y, s)), 4)
                if rec["val_pr_auc"] > best and checkpoint_dir is not None:
                    best = rec["val_pr_auc"]
                    self._save_weights(Path(checkpoint_dir))
                    rec["saved"] = True
            self.history.append(rec)
            print(rec, flush=True)
        self.meta.extra["history"] = self.history
        if checkpoint_dir is not None and best >= 0:
            self._load_weights(Path(checkpoint_dir))  # возвращаем лучшую по validation эпоху
        return self

    def predict_scores(self, texts):
        import torch

        self.model.eval()
        out = np.zeros(len(texts))
        with torch.no_grad():
            for i, enc in self._batches(list(texts), 16):
                probs = torch.softmax(self.model(**enc).logits, dim=-1)[:, 1]
                out[i:i + len(probs)] = probs.cpu().numpy()
        return out

    def _save_weights(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "hf")
        self.tokenizer.save_pretrained(path / "hf")

    def _load_weights(self, path: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.tokenizer = AutoTokenizer.from_pretrained(path / "hf")
        self.model = AutoModelForSequenceClassification.from_pretrained(path / "hf").to(self._device())
        self.model.eval()
