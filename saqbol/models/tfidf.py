"""Модели №2 и №3: TF-IDF + логистическая регрессия.

analyzer="word"      — только словесные n-граммы (1–2);
analyzer="wordchar"  — словесные n-граммы + символьные n-граммы (2–5, char_wb).
Символьные n-граммы устойчивее к опечаткам, морфологии казахского и смешению языков.
TF-IDF обучается только на train (fit вызывается только на train-выборке).
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from saqbol.models.base import ModelMeta, ScamClassifier


def preprocess(text: str) -> str:
    """Нормализация: регистр, ё, маскирование ссылок, телефонов и чисел.

    Маскирование уменьшает запоминание конкретных сумм/номеров и делает модель
    менее чувствительной к персональным данным.
    """
    t = (text or "").lower().replace("ё", "е")
    t = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", t)
    t = re.sub(r"\+?\d[\d\s\-()]{8,}\d", " PHONETOKEN ", t)
    t = re.sub(r"\*\d{4}", " CARDTOKEN ", t)
    t = re.sub(r"\d+", " 0 ", t)
    return re.sub(r"\s+", " ", t).strip()


class TfidfLogReg(ScamClassifier):
    model_type = "tfidf_lr"

    def __init__(self, analyzer: str = "word", C: float = 4.0, seed: int = 42, meta: ModelMeta | None = None):
        params = {"analyzer": analyzer, "C": C, "class_weight": "balanced",
                  "word_ngram_range": [1, 2], "char_ngram_range": [2, 5], "min_df": 1,
                  "sublinear_tf": True, "max_iter": 2000}
        name = f"tfidf_{analyzer}"
        super().__init__(meta or ModelMeta(model_type="tfidf_lr", name=name, seed=seed, params=params))
        self.pipeline: Pipeline | None = None

    def _build(self) -> Pipeline:
        p = self.meta.params
        word = TfidfVectorizer(preprocessor=preprocess, analyzer="word", ngram_range=tuple(p["word_ngram_range"]),
                               min_df=p["min_df"], sublinear_tf=p["sublinear_tf"], token_pattern=r"(?u)\b\w+\b")
        if p["analyzer"] == "wordchar":
            char = TfidfVectorizer(preprocessor=preprocess, analyzer="char_wb",
                                   ngram_range=tuple(p["char_ngram_range"]), min_df=p["min_df"],
                                   sublinear_tf=p["sublinear_tf"])
            features = FeatureUnion([("word", word), ("char", char)])
        else:
            features = word
        clf = LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=p["max_iter"],
                                 random_state=self.meta.seed)
        return Pipeline([("tfidf", features), ("clf", clf)])

    def fit(self, texts, y):
        self.pipeline = self._build()
        self.pipeline.fit(texts, y)
        return self

    def predict_scores(self, texts):
        if self.pipeline is None:
            raise RuntimeError("модель не обучена")
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def top_features(self, k: int = 15) -> dict:
        feats = self.pipeline.named_steps["tfidf"].get_feature_names_out()
        coef = self.pipeline.named_steps["clf"].coef_[0]
        order = np.argsort(coef)
        return {"scam": [(feats[i], round(float(coef[i]), 3)) for i in order[::-1][:k]],
                "not_scam": [(feats[i], round(float(coef[i]), 3)) for i in order[:k]]}

    def _save_weights(self, path: Path) -> None:
        joblib.dump(self.pipeline, path / "pipeline.joblib")

    def _load_weights(self, path: Path) -> None:
        self.pipeline = joblib.load(path / "pipeline.joblib")
