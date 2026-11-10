"""EDA: классы, языки, схемы, длины, источники, пропуски, дубликаты, статусы разметки."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from abayla.data.prepare import _norm
from abayla.data.schema import COLUMNS, validate


def run_eda(dataset_path: str | Path, out_dir: str | Path) -> dict:
    df = pd.read_csv(dataset_path, dtype=str, keep_default_na=False, na_values=[""])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = validate(df)
    df["n_chars"] = df["text"].fillna("").str.len()
    df["n_words"] = df["text"].fillna("").str.split().str.len()
    df["has_url"] = df["text"].fillna("").str.contains(r"https?://|www\.", regex=True)

    summary = {
        "rows": len(df),
        "validation_errors": rep.errors,
        "validation_warnings": rep.warnings,
        "missing_per_column": {c: int(df[c].isna().sum()) for c in COLUMNS if c in df},
        "label": df["label"].value_counts().to_dict(),
        "language": df["language"].value_counts().to_dict(),
        "language_by_label": pd.crosstab(df["language"], df["label"]).to_dict(),
        "scheme": df["scheme"].value_counts().to_dict(),
        "source_type": df["source_type"].value_counts().to_dict(),
        "annotation_status": df["annotation_status"].value_counts().to_dict(),
        "template_groups": int(df["template_group"].nunique()),
        "rows_per_group_mean": round(len(df) / max(1, df["template_group"].nunique()), 2),
        "exact_duplicates": int(df["text"].str.strip().str.lower().duplicated().sum()),
        "normalized_duplicates": int(df["text"].map(_norm).duplicated().sum()),
        "chars_by_label": df.groupby("label")["n_chars"].describe().round(1).to_dict(orient="index"),
        "words_by_label": df.groupby("label")["n_words"].describe().round(1).to_dict(orient="index"),
        "share_with_url_by_label": df.groupby("label")["has_url"].mean().round(3).to_dict(),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    df["label"].value_counts().plot.bar(ax=axes[0, 0], color=["#4C72B0", "#DD8452"])
    axes[0, 0].set_title("Классы")
    pd.crosstab(df["language"], df["label"]).plot.bar(ax=axes[0, 1])
    axes[0, 1].set_title("Языки по классам")
    df[df["label"] == "scam"]["scheme"].value_counts().plot.barh(ax=axes[1, 0], color="#DD8452")
    axes[1, 0].set_title("Схемы (только scam)")
    for label, part in df.groupby("label"):
        axes[1, 1].hist(part["n_chars"], bins=20, alpha=0.6, label=label)
    axes[1, 1].set_title("Длина сообщения, символы")
    axes[1, 1].legend()
    fig.suptitle(f"EDA: {Path(dataset_path).name} (n={len(df)})")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_overview.png", dpi=120)
    plt.close(fig)

    (out_dir / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# EDA: {Path(dataset_path).name}",
        "",
        f"Строк: {summary['rows']}, групп шаблонов: {summary['template_groups']} "
        f"(в среднем {summary['rows_per_group_mean']} текста на группу).",
        f"Ошибки валидации: {summary['validation_errors'] or 'нет'}; предупреждения: {summary['validation_warnings'] or 'нет'}.",
        f"Пропуски: {summary['missing_per_column']}.",
        f"Точные дубликаты: {summary['exact_duplicates']}; после нормализации цифр и пунктуации: {summary['normalized_duplicates']}.",
        "",
        "| Срез | Значения |",
        "|---|---|",
        f"| Классы | {summary['label']} |",
        f"| Языки | {summary['language']} |",
        f"| Схемы | {summary['scheme']} |",
        f"| Источники | {summary['source_type']} |",
        f"| Статус разметки | {summary['annotation_status']} |",
        f"| Доля сообщений со ссылкой | {summary['share_with_url_by_label']} |",
        "",
        "Длина в символах по классам (mean / 50% / max):",
    ]
    for label, st in summary["chars_by_label"].items():
        lines.append(f"- {label}: {st['mean']} / {st['50%']} / {st['max']}")
    lines += ["", "![EDA](eda_overview.png)"]
    (out_dir / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demo/abayla_demo_v1.csv")
    ap.add_argument("--out", default="reports/eda")
    args = ap.parse_args()
    s = run_eda(args.data, args.out)
    print(json.dumps({k: s[k] for k in ("rows", "label", "language", "annotation_status")}, ensure_ascii=False))
