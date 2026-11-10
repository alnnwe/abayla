"""Модель №1: простая система правил на интерпретируемых признаках.

Score = сигмоида от взвешенной суммы признаков. Веса заданы вручную (не обучаются),
fit() ничего не делает — это честный нижний ориентир для сравнения.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from saqbol.features.signals import SIGNALS, detect_signals
from saqbol.models.base import ModelMeta, ScamClassifier


class RuleClassifier(ScamClassifier):
    model_type = "rules"

    def __init__(self, bias: float = 2.0, meta: ModelMeta | None = None):
        super().__init__(meta or ModelMeta(model_type="rules", name="rules", params={"bias": bias}))
        self.bias = self.meta.params.get("bias", bias)

    def fit(self, texts, y):
        return self

    def raw_score(self, text: str) -> float:
        res = detect_signals(text)
        return sum(SIGNALS[s].weight for s in res.signals) - self.bias

    def predict_scores(self, texts):
        return np.array([1.0 / (1.0 + math.exp(-self.raw_score(t))) for t in texts])

    def _save_weights(self, path: Path) -> None:
        weights = {k: v.weight for k, v in SIGNALS.items()}
        (path / "rule_weights.json").write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_weights(self, path: Path) -> None:
        self.bias = self.meta.params.get("bias", 2.0)
