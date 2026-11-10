"""Формат датасета SaqBol и его валидация."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

COLUMNS = [
    "id",
    "text",
    "label",
    "scheme",
    "language",
    "source_type",
    "source_id",
    "template_group",
    "annotation_status",
]

LABELS = {"scam", "not_scam"}
LABEL_TO_INT = {"not_scam": 0, "scam": 1}

# для not_scam схема = "none"
SCHEMES = {
    "bank_police_impersonation",  # имитация банка или полиции
    "phishing",  # фишинг (ссылка/форма для ввода данных)
    "relative_in_trouble",  # родственник в беде
    "fake_ad",  # фальшивое объявление / предоплата
    "investment",  # инвестиционная схема
    "other_scheme",  # другая схема
    "none",
}

LANGUAGES = {"ru", "kk", "mixed"}

# real            — реальное сообщение, полученное с согласия и обезличенное
# reconstruction  — реконструкция по публичному описанию схемы (указать источник в source_id)
# synthetic       — синтетический пример, написанный человеком или сгенерированный
# synthetic_demo  — демонстрационный набор этого репозитория (НЕ доказательство качества)
SOURCE_TYPES = {"real", "reconstruction", "synthetic", "synthetic_demo"}

# verified — проверено двумя разметчиками / согласовано
# pending  — ещё не проверено
# disputed — разметчики не согласны; исключается из обучения до проверки
ANNOTATION_STATUSES = {"verified", "pending", "disputed"}
TRAINABLE_STATUSES = {"verified"}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(df: pd.DataFrame) -> ValidationReport:
    rep = ValidationReport()
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        rep.errors.append(f"нет обязательных колонок: {missing}")
        return rep

    for col in COLUMNS:
        n_null = int(df[col].isna().sum())
        if n_null:
            rep.errors.append(f"колонка {col}: {n_null} пустых значений")

    if df["id"].duplicated().any():
        rep.errors.append(f"повторяющиеся id: {df.loc[df['id'].duplicated(), 'id'].tolist()[:5]}")

    empty_text = df["text"].fillna("").astype(str).str.strip().eq("")
    if empty_text.any():
        rep.errors.append(f"пустой текст в {int(empty_text.sum())} строках")

    checks = [
        ("label", LABELS),
        ("scheme", SCHEMES),
        ("language", LANGUAGES),
        ("source_type", SOURCE_TYPES),
        ("annotation_status", ANNOTATION_STATUSES),
    ]
    for col, allowed in checks:
        bad = ~df[col].isin(allowed)
        if bad.any():
            rep.errors.append(f"колонка {col}: недопустимые значения {sorted(df.loc[bad, col].astype(str).unique())[:5]}")

    inconsistent = (df["label"] == "not_scam") & (df["scheme"] != "none")
    if inconsistent.any():
        rep.errors.append(f"not_scam со схемой != none: {int(inconsistent.sum())} строк")
    no_scheme = (df["label"] == "scam") & (df["scheme"] == "none")
    if no_scheme.any():
        rep.errors.append(f"scam без схемы: {int(no_scheme.sum())} строк")

    # одна группа шаблонов не должна содержать оба класса — это признак ошибки разметки
    mixed = df.groupby("template_group")["label"].nunique()
    if (mixed > 1).any():
        rep.warnings.append(f"группы с обоими классами: {mixed[mixed > 1].index.tolist()[:5]}")

    dup_text = df["text"].str.strip().str.lower().duplicated()
    if dup_text.any():
        rep.warnings.append(f"точные дубликаты текста: {int(dup_text.sum())}")
    return rep


def trainable(df: pd.DataFrame) -> pd.DataFrame:
    """Только проверенные примеры. disputed/pending исключаются до проверки."""
    return df[df["annotation_status"].isin(TRAINABLE_STATUSES)].copy()
