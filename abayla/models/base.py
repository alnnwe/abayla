"""Общий интерфейс классификатора. Бот и сервис зависят только от него,
поэтому baseline заменяется на трансформер сменой каталога модели (ABAYLA_MODEL_DIR)."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Thresholds:
    """Пороги подобраны на validation.

    binary  — порог для бинарных метрик (precision/recall класса scam);
    low     — ниже этого score ответ «низкий риск»;
    high    — от этого score ответ «высокий риск»; между low и high — «неопределённость».
    Score — некалиброванная оценка модели, а не вероятность мошенничества.
    """

    binary: float = 0.5
    low: float = 0.3
    high: float = 0.7
    preliminary: bool = True
    selection_note: str = ""


@dataclass
class ModelMeta:
    model_type: str
    name: str
    data_version: str = "unknown"
    demo_data: bool = True
    seed: int = 42
    params: dict = field(default_factory=dict)
    thresholds: Thresholds = field(default_factory=Thresholds)
    train_time_s: float | None = None
    hardware: str = ""
    created_at: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "ModelMeta":
        d = json.loads(raw)
        d["thresholds"] = Thresholds(**d.get("thresholds", {}))
        return cls(**d)


class ScamClassifier(ABC):
    model_type: str = "base"

    def __init__(self, meta: ModelMeta | None = None):
        self.meta = meta or ModelMeta(model_type=self.model_type, name=self.model_type)

    @abstractmethod
    def fit(self, texts: list[str], y: np.ndarray) -> "ScamClassifier": ...

    @abstractmethod
    def predict_scores(self, texts: list[str]) -> np.ndarray:
        """Возвращает score в [0, 1]; больше = больше похоже на мошенничество."""

    @abstractmethod
    def _save_weights(self, path: Path) -> None: ...

    @abstractmethod
    def _load_weights(self, path: Path) -> None: ...

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._save_weights(path)
        (path / "meta.json").write_text(self.meta.to_json(), encoding="utf-8")
        return path


def load_model(path: str | Path) -> ScamClassifier:
    path = Path(path)
    meta_file = path / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"Модель не найдена в {path}. Сначала обучите её: python -m abayla.training.train"
        )
    meta = ModelMeta.from_json(meta_file.read_text(encoding="utf-8"))
    if meta.model_type == "rules":
        from abayla.models.rules import RuleClassifier as cls
    elif meta.model_type == "tfidf_lr":
        from abayla.models.tfidf import TfidfLogReg as cls
    elif meta.model_type == "transformer":
        from abayla.models.transformer import TransformerClassifier as cls
    else:
        raise ValueError(f"неизвестный тип модели: {meta.model_type}")
    model = cls(meta=meta)
    model._load_weights(path)
    return model
