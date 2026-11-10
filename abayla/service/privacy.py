"""Техническое обезличивание перед внешними API и в логах.

Это техническая мера снижения риска, а НЕ подтверждение соответствия законодательству
о персональных данных: для такого вывода нужна отдельная юридическая проверка.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
from urllib.parse import urlsplit

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'«»]+")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)")
IIN = re.compile(r"(?<!\d)\d{12}(?!\d)")
PHONE = re.compile(r"(?<!\d)(?:\+7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)")
MASKED_CARD = re.compile(r"\*\d{4}\b")
CODE = re.compile(r"(?<![\d*])\d{4,8}(?![\d])(?!\s*(?:тенге|теңге|тг|₸|kzt))", re.IGNORECASE)


def _url_host(m: re.Match) -> str:
    raw = m.group(0)
    raw = raw if raw.lower().startswith("http") else "http://" + raw
    host = (urlsplit(raw).hostname or "?").lower()
    return f"[ССЫЛКА:{host}]"


def anonymize(text: str) -> str:
    t = text or ""
    t = URL.sub(_url_host, t)
    t = EMAIL.sub("[EMAIL]", t)
    t = CARD.sub("[КАРТА]", t)
    t = IIN.sub("[ИИН]", t)
    t = PHONE.sub("[ТЕЛЕФОН]", t)
    t = MASKED_CARD.sub("[КАРТА]", t)
    t = CODE.sub("[КОД]", t)
    return t


def text_fingerprint(secret: str, scope: str, text: str) -> str:
    """HMAC от нормализованного текста: позволяет распознать повтор без хранения текста."""
    norm = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return hmac.new(secret.encode(), f"{scope}:{norm}".encode(), hashlib.sha256).hexdigest()[:32]


class RedactingFilter(logging.Filter):
    """Последний рубеж: даже если в лог случайно попадёт строка с цифрами или токеном — маскируем."""

    TOKEN = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")  # формат токена Telegram-бота

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        msg = self.TOKEN.sub("[BOT_TOKEN]", msg)
        msg = anonymize(msg)
        record.msg, record.args = msg, ()
        return True


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)  # не логировать содержимое апдейтов
