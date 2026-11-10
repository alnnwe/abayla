"""Подбор порогов на validation (никогда на test)."""
from __future__ import annotations

import numpy as np

from saqbol.models.base import Thresholds


def _pr_at(y: np.ndarray, s: np.ndarray, t: float) -> tuple[float, float]:
    pred = s >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def _candidates(s: np.ndarray) -> np.ndarray:
    u = np.unique(s)
    mids = (u[:-1] + u[1:]) / 2 if len(u) > 1 else u
    return np.unique(np.concatenate([mids, [u.min()]]))


def select_thresholds(
    y_val: np.ndarray,
    s_val: np.ndarray,
    target_recall: float = 0.90,
    low_recall: float = 0.98,
    high_precision: float = 0.95,
    min_positives_for_final: int = 100,
) -> Thresholds:
    """binary: максимальная precision при recall >= target_recall (при равенстве — больший F1).
    low:  наибольший порог, при котором recall >= low_recall (почти все scam как минимум «неопределённость»).
    high: наименьший порог >= binary, при котором precision >= high_precision.
    Кандидаты — средние точки между соседними score, чтобы порог не «прилипал» к одному примеру.
    """
    cands = _candidates(s_val)
    best_t, best_key = float(cands.min()), (-1.0, -1.0)
    for t in cands:
        p, r = _pr_at(y_val, s_val, t)
        if r >= target_recall:
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            if (p, f1) > best_key:
                best_key, best_t = (p, f1), float(t)
    t_bin = best_t

    low_ok = [float(t) for t in cands if _pr_at(y_val, s_val, t)[1] >= low_recall]
    t_low = min(max(low_ok) if low_ok else float(cands.min()), t_bin)

    high_ok = [float(t) for t in cands if t >= t_bin and _pr_at(y_val, s_val, t)[0] >= high_precision
               and _pr_at(y_val, s_val, t)[1] > 0]
    t_high = min(high_ok) if high_ok else t_bin

    n_pos = int(np.sum(y_val == 1))
    p_bin, r_bin = _pr_at(y_val, s_val, t_bin)
    note = (f"val: n={len(y_val)}, scam={n_pos}; при binary={t_bin:.3f} precision={p_bin:.3f}, recall={r_bin:.3f}; "
            f"критерии: recall>={target_recall}, low: recall>={low_recall}, high: precision>={high_precision}")
    return Thresholds(binary=round(t_bin, 4), low=round(t_low, 4), high=round(t_high, 4),
                      preliminary=n_pos < min_positives_for_final, selection_note=note)
