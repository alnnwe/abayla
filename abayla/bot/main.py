"""Запуск Telegram-бота: python -m abayla.bot.main  (нужен BOT_TOKEN в окружении или .env)."""
from __future__ import annotations

import asyncio
import logging
import os
import ssl
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from abayla.bot.handlers import router
from abayla.bot.logic import AbaylaCore
from abayla.config import get_settings
from abayla.service.analyzer import Analyzer
from abayla.service.privacy import setup_logging
from abayla.storage.db import Storage

log = logging.getLogger("abayla")


def build_session() -> AiohttpSession | None:
    """Необязательно: свой набор корневых сертификатов (ABAYLA_CA_BUNDLE) для сетей с перехватом TLS
    (корпоративные/университетские прокси). По умолчанию aiogram использует certifi."""
    ca = os.getenv("ABAYLA_CA_BUNDLE")
    if not ca:
        return None
    session = AiohttpSession()
    session._connector_init["ssl"] = ssl.create_default_context(cafile=ca)
    return session


def build_dispatcher(core: AbaylaCore) -> Dispatcher:
    # MemoryStorage: состояние диалога только в оперативной памяти процесса, на диск не пишется
    dp = Dispatcher(storage=MemoryStorage(), core=core)
    dp.include_router(router)
    return dp


async def run() -> None:
    settings = get_settings()
    setup_logging()
    if not settings.bot_token:
        print("BOT_TOKEN не задан. Для проверки без Telegram используйте: python -m abayla.cli repl", file=sys.stderr)
        sys.exit(2)
    analyzer = Analyzer.from_dir(settings.model_dir)
    if analyzer.model.meta.demo_data:
        log.warning("ДЕМО-РЕЖИМ: модель %s обучена на демонстрационных данных", analyzer.model.meta.name)
    if settings.hash_secret == "dev-only-change-me":
        log.warning("ABAYLA_HASH_SECRET не задан — используется значение для разработки")
    core = AbaylaCore(Storage(settings.db_path), analyzer, settings)
    dp = build_dispatcher(core)
    bot = Bot(token=settings.bot_token, session=build_session())
    try:
        await _start(bot, dp, analyzer)
    finally:
        await bot.session.close()


async def _start(bot: Bot, dp: Dispatcher, analyzer: Analyzer) -> None:
    await bot.set_my_commands([
        BotCommand(command="start", description="Начать / Бастау"),
        BotCommand(command="help", description="Помощь / Көмек"),
        BotCommand(command="language", description="Язык / Тіл"),
        BotCommand(command="settings", description="Настройки / Баптаулар"),
        BotCommand(command="privacy", description="Данные / Деректер"),
    ])
    log.info("bot started, model=%s", analyzer.model.meta.name)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(run())
