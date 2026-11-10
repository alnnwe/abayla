"""Загрузка, импорт пользовательского CSV, разбиение по группам и аугментация (только train)."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from saqbol.data.schema import COLUMNS, LABEL_TO_INT, trainable, validate

SPLIT_NAMES = ("train", "val", "test")


def load_dataset(path: str | Path, strict: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    rep = validate(df)
    if strict and not rep.ok:
        raise ValueError("Датасет не прошёл проверку:\n- " + "\n- ".join(rep.errors))
    return df


def import_user_csv(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    defaults: dict | None = None,
    column_map: dict | None = None,
) -> pd.DataFrame:
    """Импорт произвольного CSV в формат SaqBol.

    Недостающие колонки заполняются значениями из ``defaults``. Если у примера нет
    template_group, создаётся группа по нормализованному тексту (точные дубликаты окажутся
    в одной группе; переводы и перефразировки нужно группировать вручную, см. docs/annotation_guide.md).
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    column_map = column_map or {}
    raw = raw.rename(columns={v: k for k, v in column_map.items()})
    raw = raw.rename(columns={text_col: "text", label_col: "label"})
    defaults = {
        "scheme": "other_scheme",
        "language": "ru",
        "source_type": "real",
        "source_id": Path(path).stem,
        "annotation_status": "pending",
        **(defaults or {}),
    }
    out = pd.DataFrame()
    out["text"] = raw["text"].astype(str).str.strip()
    out["label"] = raw["label"].astype(str).str.strip().str.lower().replace(
        {"1": "scam", "0": "not_scam", "fraud": "scam", "legit": "not_scam", "ham": "not_scam"}
    )
    for col in ("scheme", "language", "source_type", "source_id", "annotation_status"):
        out[col] = raw[col] if col in raw.columns else defaults[col]
    out.loc[out["label"] == "not_scam", "scheme"] = "none"
    if "template_group" in raw.columns:
        out["template_group"] = raw["template_group"]
    else:
        out["template_group"] = [
            "auto_" + hashlib.sha1(_norm(t).encode()).hexdigest()[:12] for t in out["text"]
        ]
    out["id"] = raw["id"] if "id" in raw.columns else [f"{Path(path).stem}_{i:05d}" for i in range(len(out))]
    return out[COLUMNS]


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+", "0", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def group_split(
    df: pd.DataFrame, seed: int = 42, fractions=(0.70, 0.15, 0.15)
) -> pd.DataFrame:
    """Разбиение по template_group, стратифицированное по классу на уровне групп.

    Все тексты одной группы (оригинал, перевод, перефразировка) попадают в одну выборку.
    """
    assert abs(sum(fractions) - 1.0) < 1e-6
    rng = random.Random(seed)
    group_label = df.groupby("template_group")["label"].agg(lambda s: s.mode().iat[0])
    assignment: dict[str, str] = {}
    for label in sorted(group_label.unique()):
        groups = sorted(group_label[group_label == label].index.tolist())
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * fractions[0]))
        n_val = int(round(n * fractions[1]))
        n_val = max(1, n_val) if n >= 3 else n_val
        for i, grp in enumerate(groups):
            if i < n_train:
                assignment[grp] = "train"
            elif i < n_train + n_val:
                assignment[grp] = "val"
            else:
                assignment[grp] = "test"
    out = df.copy()
    out["split"] = out["template_group"].map(assignment)
    return out


def check_no_group_leakage(split_df: pd.DataFrame) -> None:
    per_group = split_df.groupby("template_group")["split"].nunique()
    leaked = per_group[per_group > 1]
    if len(leaked):
        raise AssertionError(f"группы в нескольких выборках: {leaked.index.tolist()[:5]}")


_TYPO_RNG = None


def augment_train(train_df: pd.DataFrame, seed: int = 42, copies: int = 1) -> pd.DataFrame:
    """Простая аугментация ТОЛЬКО для train (вызывается после разбиения).

    Имитирует небрежный ввод: нижний регистр, удаление пунктуации, случайные пропуски букв.
    Новые строки получают ту же template_group и пометку augmented=1.
    """
    rng = random.Random(seed)
    rows = []
    for _, row in train_df.iterrows():
        for c in range(copies):
            text = row["text"].lower()
            text = re.sub(r"[!?.,:;«»\"]", "", text)
            chars = list(text)
            for i in range(len(chars)):
                if chars[i].isalpha() and rng.random() < 0.03:
                    chars[i] = ""
            new = dict(row)
            new["text"] = "".join(chars)
            new["id"] = f"{row['id']}_aug{c}"
            new["augmented"] = 1
            rows.append(new)
    base = train_df.copy()
    base["augmented"] = 0
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)


def prepare_splits(
    dataset_path: str | Path, out_dir: str | Path, seed: int = 42, augment: bool = False
) -> dict:
    df = load_dataset(dataset_path)
    total = len(df)
    df = trainable(df)
    split_df = group_split(df, seed=seed)
    check_no_group_leakage(split_df)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"dataset": str(dataset_path), "seed": seed, "rows_total": total, "rows_trainable": len(df)}
    for name in SPLIT_NAMES:
        part = split_df[split_df["split"] == name].drop(columns=["split"])
        if name == "train" and augment:
            part = augment_train(part, seed=seed)
        part.to_csv(out_dir / f"{name}.csv", index=False)
        info[name] = {
            "rows": len(part),
            "groups": int(part["template_group"].nunique()),
            "scam": int((part["label"] == "scam").sum()),
            "not_scam": int((part["label"] == "not_scam").sum()),
        }
    info["augment"] = augment
    info["data_hash"] = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()[:16]
    (out_dir / "split_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def load_split(split_dir: str | Path, name: str) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    df = pd.read_csv(Path(split_dir) / f"{name}.csv", dtype=str, keep_default_na=False)
    y = df["label"].map(LABEL_TO_INT).to_numpy()
    return df["text"].tolist(), y, df
