"""Локальный интерфейс проверки без Telegram-токена.

    python -m saqbol.cli check "Продиктуйте код из SMS"
    python -m saqbol.cli check --lang kk "Кодты айтыңыз"
    python -m saqbol.cli repl
    python -m saqbol.cli scenarios      # проверочные пары «уведомление с кодом / просьба кода» и др.
"""
from __future__ import annotations

import argparse
import sys

from saqbol.explain.service import explain
from saqbol.service.analyzer import Analyzer, EmptyTextError

SCENARIOS = [
    ("Ваш код входа: 4821. Никому его не сообщайте.", "low"),
    ("Продиктуйте код из SMS, чтобы отменить операцию.", "high"),
    ("Кіру кодыңыз: 4821. Оны ешкімге айтпаңыз.", "low"),
    ("SMS-тегі кодты айтыңыз, операцияны тоқтатамыз.", "high"),
    ("Мам, я доехал, всё хорошо. Вечером позвоню.", "low"),
    ("Мама, это я, у меня новый номер. Попал в аварию, срочно переведи деньги.", "high"),
    ("Демобанк предупреждает: сотрудники банка никогда не просят коды из SMS.", "low"),
    ("Переведите сбережения на безопасный счёт, это служба безопасности банка.", "high"),
    ("Перейдите по ссылке http://demobank-verify.example.com/login и подтвердите данные карты.", "high"),
    ("Скидка 20% на все товары до конца недели.", "low"),
]


def format_result(analyzer: Analyzer, text: str, lang: str, verbose: bool = True) -> str:
    try:
        a = analyzer.analyze(text)
    except EmptyTextError:
        return "Пустой текст." if lang == "ru" else "Мәтін бос."
    out = explain(a, lang, original_text=text).render(lang)
    if verbose:
        th = analyzer.model.meta.thresholds
        out += (f"\n\n[debug] model={a.model_name} score={a.score:.3f} (не вероятность) "
                f"thresholds low={th.low} high={th.high}{' (предварительные)' if th.preliminary else ''} "
                f"signals={a.signals} {a.latency_ms} ms")
        for r in a.url_reports:
            out += f"\n[debug] url {r.host} flags={r.flags} score={r.score}"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="saqbol")
    ap.add_argument("--model", default=None, help="каталог модели (по умолчанию SAQBOL_MODEL_DIR)")
    ap.add_argument("--lang", default="ru", choices=["ru", "kk"])
    ap.add_argument("--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("text", nargs="+")
    sub.add_parser("repl")
    sub.add_parser("scenarios")
    args = ap.parse_args(argv)
    analyzer = Analyzer.from_dir(args.model)
    if analyzer.model.meta.demo_data:
        print("=== ДЕМО-РЕЖИМ: модель обучена на демонстрационных синтетических данных ===\n")
    if args.cmd == "check":
        print(format_result(analyzer, " ".join(args.text), args.lang, not args.quiet))
    elif args.cmd == "repl":
        print("Вставьте текст и нажмите Enter (пустая строка — выход).")
        for line in sys.stdin:
            if not line.strip():
                break
            print(format_result(analyzer, line, args.lang, not args.quiet), "\n")
    else:
        ok = 0
        for text, expected in SCENARIOS:
            a = analyzer.analyze(text)
            passed = (a.level == "low") == (expected == "low")
            ok += passed
            print(f"[{'OK ' if passed else 'ERR'}] ожидалось {expected:<9} получено {a.level:<9} "
                  f"score={a.score:.2f} | {text[:60]}")
        print(f"\n{ok}/{len(SCENARIOS)} сценариев совпали по признаку «низкий риск / не низкий»")
        return 0 if ok == len(SCENARIOS) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
