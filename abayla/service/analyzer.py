"""Сервис анализа: модель → уровень риска; признаки → объяснение.

Итоговый уровень определяет классификатор (пороги с validation). Единственная поправка
политики безопасности: если найден «сильный» признак (просьба кода, данных карты,
«безопасный счёт», удалённый доступ, ИИН/кодовое слово), ответ не может быть «низкий риск» —
он поднимается до «неопределённости». LLM на уровень не влияет.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from abayla.config import get_settings
from abayla.features.signals import SIGNALS, detect_signals
from abayla.models.base import ScamClassifier, load_model

LEVELS = ("low", "uncertain", "high")


class EmptyTextError(ValueError):
    pass


@dataclass
class Analysis:
    level: str
    score: float
    signals: list[str]
    url_reports: list = field(default_factory=list)
    model_name: str = ""
    demo: bool = False
    thresholds_preliminary: bool = True
    overridden: bool = False
    override_reason: str = ""
    truncated: bool = False
    needs_clarification: bool = False
    latency_ms: float = 0.0

    @property
    def risk_signals(self) -> list[str]:
        return [s for s in self.signals if SIGNALS[s].weight > 0 and s != "link_present"]


class Analyzer:
    def __init__(self, model: ScamClassifier, max_chars: int | None = None, apply_policy: bool = True):
        self.model = model
        self.max_chars = max_chars or get_settings().max_text_chars
        self.apply_policy = apply_policy

    @classmethod
    def from_dir(cls, path: str | Path | None = None) -> "Analyzer":
        return cls(load_model(path or get_settings().model_dir))

    def level_for(self, score: float) -> str:
        th = self.model.meta.thresholds
        if score >= th.high:
            return "high"
        if score < th.low:
            return "low"
        return "uncertain"

    def analyze(self, text: str) -> Analysis:
        t0 = time.perf_counter()
        text = (text or "").strip()
        if not text:
            raise EmptyTextError("пустой текст")
        truncated = len(text) > self.max_chars
        if truncated:
            text = text[: self.max_chars]
        score = float(self.model.predict_scores([text])[0])
        sig = detect_signals(text)
        level = self.level_for(score)
        overridden, reason = False, ""
        short = len(text.split()) < 8
        risk = [s for s in sig.signals if SIGNALS[s].weight > 0 and s != "link_present"]
        if self.apply_policy and level == "low":
            if sig.strong:
                level, overridden, reason = "uncertain", True, "strong_signal"
            elif short and risk:
                # короткий текст с признаком риска: контекста мало — задаём один уточняющий вопрос
                level, overridden, reason = "uncertain", True, "short_with_signal"
        needs_clarification = level == "uncertain" and short
        return Analysis(
            level=level,
            score=score,
            signals=sig.signals,
            url_reports=sig.url_reports,
            model_name=self.model.meta.name,
            demo=self.model.meta.demo_data,
            thresholds_preliminary=self.model.meta.thresholds.preliminary,
            overridden=overridden,
            override_reason=reason,
            truncated=truncated,
            needs_clarification=needs_clarification,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
