"""Анализ структуры ссылок БЕЗ открытия страниц.

Отсутствие подозрительных признаков не доказывает безопасность ссылки: новый мошеннический
домен может выглядеть совершенно обычно. Внешние сервисы репутации не подключены по умолчанию
(см. docs/responsible_ai.md, раздел про URL).
"""
from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "url_rules.json"

URL_RE = re.compile(
    r"""(?xi)
    (?:https?://[^\s<>"'«»]+)                      # со схемой
    | (?:www\.[^\s<>"'«»]+)                         # www.
    | (?:\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.[a-z]{2,24}/[^\s<>"'«»]*)  # домен/путь без схемы
    """
)

# двухуровневые публичные суффиксы, нужные для выделения «регистрируемого» домена
TWO_LEVEL_SUFFIXES = {"com.kz", "org.kz", "gov.kz", "edu.kz", "co.uk", "com.ru", "net.kz", "co.kz"}


@lru_cache(maxsize=1)
def load_rules() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@dataclass
class UrlReport:
    url: str
    host: str
    registered_domain: str
    flags: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def suspicious(self) -> bool:
        return self.score >= load_rules()["suspicious_score"]


FLAG_WEIGHTS = {
    "ip_host": 2.0,
    "userinfo_at": 2.5,
    "punycode": 1.5,
    "shortener": 1.0,
    "suspicious_tld": 1.0,
    "brand_lookalike": 2.5,
    "many_subdomains": 1.0,
    "many_hyphens": 0.7,
    "sensitive_keywords": 1.0,
    "apk_download": 2.0,
    "no_https": 0.5,
    "unusual_port": 1.0,
    "very_long": 0.5,
}


def extract_urls(text: str) -> list[str]:
    urls = [m.group(0).rstrip(".,;:!?)") for m in URL_RE.finditer(text or "")]
    return list(dict.fromkeys(urls))


def registered_domain(host: str) -> str:
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in TWO_LEVEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def analyze_url(url: str) -> UrlReport:
    rules = load_rules()
    raw = url if re.match(r"(?i)^https?://", url) else "http://" + url
    parts = urlsplit(raw)
    netloc = parts.netloc
    flags: list[str] = []
    if "@" in netloc:
        flags.append("userinfo_at")
    host = (parts.hostname or "").lower().rstrip(".")
    reg = host
    try:
        ipaddress.ip_address(host)
        flags.append("ip_host")
    except ValueError:
        reg = registered_domain(host)
        labels = host.split(".")
        if any(lbl.startswith("xn--") for lbl in labels):
            flags.append("punycode")
        if reg in rules["shorteners"]:
            flags.append("shortener")
        if labels[-1] in rules["suspicious_tlds"]:
            flags.append("suspicious_tld")
        if len(labels) - len(reg.split(".")) >= 2:
            flags.append("many_subdomains")
        if host.count("-") >= 2:
            flags.append("many_hyphens")
        for brand, official in rules["brands"].items():
            if brand in host and reg not in official and host not in official:
                flags.append("brand_lookalike")
                break
    path_and_host = (host + parts.path + "?" + parts.query).lower()
    if any(kw in path_and_host for kw in rules["sensitive_keywords"]):
        flags.append("sensitive_keywords")
    if parts.path.lower().endswith(".apk"):
        flags.append("apk_download")
    if parts.scheme == "http" and re.match(r"(?i)^http://", url):
        flags.append("no_https")
    try:
        if parts.port and parts.port not in (80, 443):
            flags.append("unusual_port")
    except ValueError:
        flags.append("unusual_port")
    if len(url) > 100:
        flags.append("very_long")
    score = sum(FLAG_WEIGHTS[f] for f in flags)
    return UrlReport(url=url, host=host, registered_domain=reg, flags=flags, score=round(score, 2))


def analyze_text_urls(text: str) -> list[UrlReport]:
    return [analyze_url(u) for u in extract_urls(text)]
