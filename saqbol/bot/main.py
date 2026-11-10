"""Запуск Telegram-бота: python -m saqbol.bot.main  (нужен BOT_TOKEN в окружении или .env)."""
from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from saqbol.bot.handlers import router
from saqbol.bot.logic import SaqBolCore
from saqbol.config import get_settings
from saqbol.service.analyzer import Analyzer
from saqbol.service.privacy import setup_logging
from saqbol.storage.db import Storage

log = logging.getLogger("saqbol")


def build_dispatcher(core: SaqBolCore) -> Dispatcher:
    # MemoryStorage: состояние диалога только в оперативной памяти процесса, на диск не пишется
    dp = Dispatcher(storage=MemoryStorage(), core=core)
    dp.include_router(router)
    return dp


async def run() -> None:
    settings = get_settings()
    setup_logging()
    if not settings.bot_token:
        print("BOT_TOKEN не задан. Для проверки без Telegram используйте: python -m saqbol.cli repl", file=sys.stderr)
        sys.exit(2)
    analyzer = Analyzer.from_dir(settings.model_dir)
    if analyzer.model.meta.demo_data:
        log.warning("ДЕМО-РЕЖИМ: модель %s обучена на демонстрационных данных", analyzer.model.meta.name)
    if settings.hash_secret == "dev-only-change-me":
        log.warning("SAQBOL_HASH_SECRET не задан — используется значение для разработки")
    core = SaqBolCore(Storage(settings.db_path), analyzer, settings)
    dp = build_dispatcher(core)
    bot = Bot(token=settings.bot_token)
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
