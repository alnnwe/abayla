"""Выбор способа объяснения: проверенный шаблон (по умолчанию) или LLM-переформулировка.

LLM включается только при LLM_ENABLED=1 и наличии ключа. Любая ошибка, тайм-аут или
непрошедший проверку ответ → шаблон.
"""
from __future__ import annotations

from abayla.config import get_settings
from abayla.explain.templates import Explanation, template_explanation

_llm = None


def explain(analysis, lang: str, original_text: str | None = None) -> Explanation:
    base = template_explanation(analysis, lang)
    settings = get_settings()
    if not (settings.llm_enabled and settings.llm_api_key and original_text):
        return base
    global _llm
    if _llm is None:
        from abayla.explain.llm import LLMExplainer

        _llm = LLMExplainer.from_settings(settings)
    return _llm.rephrase(base, analysis, lang, original_text)
