"""Настройки проекта. Все секреты берутся только из переменных окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # .env удобен локально, но не обязателен
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    bot_token: str | None = field(default_factory=lambda: os.getenv("BOT_TOKEN") or None)
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ABAYLA_MODEL_DIR", ROOT / "artifacts" / "models" / "tfidf_wordchar"))
    )
    db_path: Path = field(default_factory=lambda: Path(os.getenv("ABAYLA_DB_PATH", ROOT / "abayla.sqlite3")))
    # секрет для HMAC-хешей (дедупликация уведомлений без хранения текста)
    hash_secret: str = field(default_factory=lambda: os.getenv("ABAYLA_HASH_SECRET", "dev-only-change-me"))
    max_text_chars: int = field(default_factory=lambda: int(os.getenv("ABAYLA_MAX_TEXT_CHARS", "4000")))
    link_ttl_minutes: int = field(default_factory=lambda: int(os.getenv("ABAYLA_LINK_TTL_MINUTES", "30")))
    notify_dedup_hours: int = field(default_factory=lambda: int(os.getenv("ABAYLA_NOTIFY_DEDUP_HOURS", "24")))
    notify_max_per_hour: int = field(default_factory=lambda: int(os.getenv("ABAYLA_NOTIFY_MAX_PER_HOUR", "3")))
    # LLM (необязательно)
    llm_enabled: bool = field(default_factory=lambda: _bool("LLM_ENABLED", False))
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-haiku-5-5"))
    llm_api_version: str = field(default_factory=lambda: os.getenv("LLM_API_VERSION", "2026-10-21"))
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "4")))
    # Голос (необязательно)
    voice_enabled: bool = field(default_factory=lambda: _bool("VOICE_ENABLED", False))
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small"))
    max_voice_seconds: int = field(default_factory=lambda: int(os.getenv("ABAYLA_MAX_VOICE_SECONDS", "120")))


def get_settings() -> Settings:
    return Settings()
