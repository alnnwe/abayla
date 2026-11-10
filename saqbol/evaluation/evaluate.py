"""Финальная оценка на test (после фиксации моделей и порогов на validation).

    python -m saqbol.evaluation.evaluate

Все модели сравниваются на ОДНОМ И ТОМ ЖЕ test (проверяется по списку id).
Результаты: reports/metrics.json, reports/results.md, reports/figures/*.png, reports/errors/*.csv
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from saqbol.config import ROOT
from saqbol.data.prepare import load_split
from saqbol.models.base import load_model
from saqbol.models.tfidf import TfidfLogReg
from saqbol.service.analyzer import Analyzer
from saqbol.training.env_info import hardware_info
from saqbol.training.thresholds import select_thresholds

TARGET_RECALL, TARGET_PRECISION, TARGET_P95_S = 0.90, 0.85, 5.0


def binary_metrics(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    pred = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)), "n_scam": int(y.sum()), "n_not_scam": int((1 - y).sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if len(set(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    return out


def group_bootstrap_ci(y, s, groups, t, n_boot=2000, seed=0) -> dict:
    """95% интервалы через бутстрэп по группам шаблонов (тексты одной группы коррелированы)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    res = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        yy, pp = y[idx], (s[idx] >= t).astype(int)
        if yy.sum() == 0:
            continue
        res["precision"].append(precision_score(yy, pp, zero_division=0))
        res["recall"].append(recall_score(yy, pp, zero_division=0))
        res["f1"].append(f1_score(yy, pp, zero_division=0))
    return {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] for k, v in res.items()}


def levels(s, th):
    return np.where(s >= th.high, "high", np.where(s < th.low, "low", "uncertain"))


def slice_table(df: pd.DataFrame, col: str) -> list[dict]:
    rows = []
    for key, part in df.groupby(col):
        y, pred = part["y"].to_numpy(), part["pred"].to_numpy()
        row = {col: key, "n": len(part), "n_scam": int(y.sum()), "n_not_scam": int((1 - y).sum())}
        if y.sum():
            row["recall"] = round(float(recall_score(y, pred)), 3)
        if (1 - y).sum():
            row["false_positive_rate"] = round(float(((pred == 1) & (y == 0)).sum() / (1 - y).sum()), 3)
        if pred.sum():
            row["precision"] = round(float(precision_score(y, pred, zero_division=0)), 3)
        rows.append(row)
    return rows


def measure_latency(model_dir: Path, texts: list[str], repeats: int = 5) -> dict:
    t0 = time.perf_counter()
    analyzer = Analyzer.from_dir(model_dir)
    load_s = time.perf_counter() - t0
    analyzer.analyze(texts[0])  # прогрев
    times = []
    for _ in range(repeats):
        for txt in texts:
            t1 = time.perf_counter()
            analyzer.analyze(txt)
            times.append(time.perf_counter() - t1)
    times_ms = np.array(times) * 1000
    return {"model_load_s": round(load_s, 3), "n_calls": len(times),
            "median_ms": round(float(np.median(times_ms)), 2), "p95_ms": round(float(np.percentile(times_ms, 95)), 2),
            "max_ms": round(float(times_ms.max()), 2),
            "scope": "Analyzer.analyze(): классификатор + признаки + анализ URL, без сети Telegram и без LLM"}


def service_levels(model_dir: Path, test_df: pd.DataFrame) -> dict:
    """Уровни, которые реально увидит пользователь (с политикой «сильный признак ≠ низкий риск»)."""
    out = {}
    for policy in (False, True):
        analyzer = Analyzer.from_dir(model_dir)
        analyzer.apply_policy = policy
        lv = [analyzer.analyze(t).level for t in test_df["text"]]
        tab = pd.crosstab(test_df["label"], pd.Series(lv, name="level", index=test_df.index))
        out["with_policy" if policy else "model_only"] = {
            lab: {lvl: int(tab.loc[lab].get(lvl, 0)) for lvl in ("low", "uncertain", "high")} for lab in tab.index
        }
    return out


def learning_curve(split_dir: Path, seeds=(0, 1, 2, 3, 4)) -> dict:
    """TF-IDF(word+char)+LR на долях train-групп; качество на VALIDATION (test не используется)."""
    _, _, tr = load_split(split_dir, "train")
    x_va, y_va, _ = load_split(split_dir, "val")
    groups = tr["template_group"].unique()
    res = {}
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        aps, f1s = [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            g = rng.choice(groups, size=max(4, int(len(groups) * frac)), replace=False)
            part = tr[tr["template_group"].isin(g)]
            if part["label"].nunique() < 2:
                continue
            y = (part["label"] == "scam").astype(int).to_numpy()
            m = TfidfLogReg(analyzer="wordchar", seed=seed).fit(part["text"].tolist(), y)
            s = m.predict_scores(x_va)
            aps.append(average_precision_score(y_va, s))
            f1s.append(f1_score(y_va, (s >= 0.5).astype(int)))
            if frac == 1.0:
                break  # на полной выборке выбор групп не меняется
        res[frac] = {"n_groups": int(max(4, int(len(groups) * frac))), "val_pr_auc_mean": round(float(np.mean(aps)), 3),
                     "val_pr_auc_std": round(float(np.std(aps)), 3), "val_f1_at_0.5_mean": round(float(np.mean(f1s)), 3)}
    return res


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], ["not_scam", "scam"])
    ax.set_yticks([0, 1], ["not_scam", "scam"])
    ax.set_xlabel("предсказано")
    ax.set_ylabel("истина")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(ROOT / "artifacts" / "splits"))
    ap.add_argument("--models", default=str(ROOT / "artifacts" / "models"))
    ap.add_argument("--extra-models", default="", help="доп. каталоги моделей через запятую (напр. аугментация)")
    ap.add_argument("--service-model", default="tfidf_wordchar")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args(argv)

    split_dir, out = Path(args.splits), Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "errors").mkdir(parents=True, exist_ok=True)
    x_te, y_te, te = load_split(split_dir, "test")
    test_ids = te["id"].tolist()
    groups = te["template_group"].to_numpy()

    model_dirs = sorted(p for p in Path(args.models).iterdir() if (p / "meta.json").exists())
    model_dirs += [Path(p) for p in args.extra_models.split(",") if p]
    report = {"hardware": hardware_info(), "test": {"n": len(te), "n_scam": int(y_te.sum()),
              "groups": int(te["template_group"].nunique()), "source_types": te["source_type"].value_counts().to_dict(),
              "languages": te["language"].value_counts().to_dict()}, "models": {}}

    # «короткий путь»: насколько хорошо класс угадывается только по длине текста
    lengths = te["text"].str.len().to_numpy()
    report["length_only_roc_auc"] = round(float(roc_auc_score(y_te, lengths)), 3)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for md in model_dirs:
        model = load_model(md)
        meta, th = model.meta, model.meta.thresholds
        name = md.name if md.parent == Path(args.models) else f"{md.parent.name}/{md.name}"
        s = model.predict_scores(x_te)
        m = binary_metrics(y_te, s, th.binary)
        m["ci95_group_bootstrap"] = group_bootstrap_ci(y_te, s, groups, th.binary)
        m["thresholds"] = {"binary": th.binary, "low": th.low, "high": th.high, "preliminary": th.preliminary}
        lv = levels(s, th)
        m["three_level"] = {lab: {L: int(((lv == L) & (y_te == yi)).sum()) for L in ("low", "uncertain", "high")}
                            for lab, yi in (("scam", 1), ("not_scam", 0))}
        m["targets_met"] = {"recall>=0.90": m["recall"] >= TARGET_RECALL,
                            "precision>=0.85": m["precision"] >= TARGET_PRECISION}
        m["targets_met"]["both_at_same_threshold"] = all(m["targets_met"].values())
        m["meta"] = {"data_version": meta.data_version, "demo_data": meta.demo_data, "seed": meta.seed,
                     "train_time_s": meta.train_time_s, "hardware": meta.hardware, "params": meta.params}
        df = te.assign(y=y_te, score=s, pred=(s >= th.binary).astype(int), level=lv)
        m["by_language"] = slice_table(df, "language")
        m["by_scheme"] = slice_table(df, "scheme")
        m["by_source_type"] = slice_table(df, "source_type")
        from saqbol.features.signals import detect_signals
        df["signals"] = [",".join(detect_signals(t).signals) for t in df["text"]]
        err = df[df["pred"] != df["y"]].assign(error=lambda d: np.where(d["y"] == 1, "FN", "FP"))
        err[["id", "error", "label", "scheme", "language", "score", "level", "signals", "text"]].sort_values(
            "score").to_csv(out / "errors" / f"errors_{name.replace('/', '_')}.csv", index=False)
        m["n_errors"] = len(err)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        plot_confusion(cm, f"{name} @ {th.binary:.3f}", out / "figures" / f"confusion_{name.replace('/', '_')}.png")
        prec, rec, _ = precision_recall_curve(y_te, s)
        ax.plot(rec, prec, label=f"{name} (AP={m.get('pr_auc', 0):.2f})")
        ax.scatter([m["recall"]], [m["precision"]], s=25)
        report["models"][name] = m
        print(f"{name:<28} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"PR-AUC={m.get('pr_auc', float('nan')):.3f} errors={m['n_errors']}")

    ax.axhline(TARGET_PRECISION, ls="--", c="grey", lw=0.8)
    ax.axvline(TARGET_RECALL, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("recall (scam)")
    ax.set_ylabel("precision (scam)")
    ax.set_title(f"Precision–recall на test (n={len(te)}, демо-данные)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "figures" / "pr_curves.png", dpi=120)
    plt.close(fig)

    svc_dir = Path(args.models) / args.service_model
    report["latency"] = {}
    for md in model_dirs:
        name = md.name if md.parent == Path(args.models) else f"{md.parent.name}/{md.name}"
        heavy = load_model(md).meta.model_type == "transformer"
        lat = measure_latency(md, x_te, repeats=1 if heavy else 5)
        lat["target_p95_met"] = lat["p95_ms"] / 1000 < TARGET_P95_S
        report["latency"][name] = lat
        print(f"latency {name}: median {lat['median_ms']} ms, p95 {lat['p95_ms']} ms")
    report["service_levels"] = service_levels(svc_dir, te)
    report["learning_curve_val"] = learning_curve(split_dir)
    lc = report["learning_curve_val"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    xs = [v["n_groups"] for v in lc.values()]
    ax.errorbar(xs, [v["val_pr_auc_mean"] for v in lc.values()], yerr=[v["val_pr_auc_std"] for v in lc.values()],
                marker="o", capsize=3, label="PR-AUC (val)")
    ax.plot(xs, [v["val_f1_at_0.5_mean"] for v in lc.values()], marker="s", label="F1@0.5 (val)")
    ax.set_xlabel("число групп шаблонов в train")
    ax.set_title("Learning curve: TF-IDF word+char + LR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "learning_curve.png", dpi=120)
    plt.close(fig)

    report["test_ids_sha"] = __import__("hashlib").sha256(",".join(test_ids).encode()).hexdigest()[:16]
    report["synthetic_vs_real"] = ("не выполнено: все примеры synthetic_demo, реальных сообщений в наборе нет"
                                   if set(te["source_type"]) == {"synthetic_demo"} else "см. by_source_type")
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(report, out / "results.md")
    return report


def _fmt_ci(ci):
    return f"[{ci[0]:.2f}; {ci[1]:.2f}]"


def write_markdown(r: dict, path: Path) -> None:
    L = ["# Результаты на test (сгенерировано saqbol.evaluation.evaluate)", "",
         "**Демонстрационные синтетические данные.** Эти числа проверяют, что конвейер работает и модели "
         "сравниваются честно. Они не являются оценкой качества на реальных сообщениях.", "",
         f"Test: n={r['test']['n']}, scam={r['test']['n_scam']}, групп={r['test']['groups']}, "
         f"языки={r['test']['languages']}. Оборудование: {r['hardware']}.", "",
         "| Модель | Порог (val) | Precision | Recall | F1 | PR-AUC | ROC-AUC | 95% ДИ recall | 95% ДИ precision | Цели P≥0.85 и R≥0.90 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in r["models"].items():
        ci = m["ci95_group_bootstrap"]
        L.append(f"| {name} | {m['thresholds']['binary']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                 f"{m['f1']:.3f} | {m.get('pr_auc', float('nan')):.3f} | {m.get('roc_auc', float('nan')):.3f} | "
                 f"{_fmt_ci(ci['recall'])} | {_fmt_ci(ci['precision'])} | "
                 f"{'да' if m['targets_met']['both_at_same_threshold'] else 'нет'} |")
    L += ["", f"ROC-AUC классификатора «только длина текста»: {r['length_only_roc_auc']} (проверка «короткого пути»).", "",
          "## Матрицы ошибок (TN FP / FN TP)", ""]
    for name, m in r["models"].items():
        L.append(f"- {name}: TN={m['tn']} FP={m['fp']} / FN={m['fn']} TP={m['tp']}")
    L += ["", "## Три уровня риска (по порогам low/high с validation)", ""]
    for name, m in r["models"].items():
        L.append(f"- {name}: scam → {m['three_level']['scam']}; not_scam → {m['three_level']['not_scam']}")
    L += ["", "## По языкам (размеры подвыборок указаны)", ""]
    for name, m in r["models"].items():
        parts = [f"{x['language']}: n={x['n']} (scam {x['n_scam']}), recall={x.get('recall', '—')}, "
                 f"FPR={x.get('false_positive_rate', '—')}" for x in m["by_language"]]
        L.append(f"- {name}: " + "; ".join(parts))
    L += ["", "## Recall по схемам", ""]
    for name, m in r["models"].items():
        parts = [f"{x['scheme']}: {x.get('recall', '—')} (n={x['n_scam']})" for x in m["by_scheme"] if x["n_scam"]]
        L.append(f"- {name}: " + "; ".join(parts))
    L += ["", "## Задержка (Analyzer.analyze: модель + признаки + URL; без сети Telegram и без LLM)", "",
          "| Модель | Вызовов | Медиана, мс | p95, мс | Загрузка, с | p95 < 5 с |", "|---|---|---|---|---|---|"]
    for name, lat in r["latency"].items():
        L.append(f"| {name} | {lat['n_calls']} | {lat['median_ms']} | {lat['p95_ms']} | {lat['model_load_s']} | "
                 f"{'да' if lat['target_p95_met'] else 'нет'} |")
    L += ["",
          "## Уровни в сервисе (с политикой «сильный признак не даёт низкий риск»)", "",
          f"- только модель: {r['service_levels']['model_only']}",
          f"- с политикой: {r['service_levels']['with_policy']}", "",
          "## Learning curve (на validation)", ""]
    for frac, v in r["learning_curve_val"].items():
        L.append(f"- {v['n_groups']} групп: PR-AUC {v['val_pr_auc_mean']} ± {v['val_pr_auc_std']}, F1@0.5 {v['val_f1_at_0.5_mean']}")
    L += ["", f"Синтетика vs реальные данные: {r['synthetic_vs_real']}.", "",
          "Ошибочные предсказания для ручного анализа: reports/errors/*.csv. Графики: reports/figures/."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
