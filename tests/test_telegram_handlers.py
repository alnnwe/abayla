"""Сквозной тест aiogram-обработчиков без сети: HTTP-сессия подменена и записывает вызовы Bot API."""
import asyncio
import datetime as dt
import itertools

import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import AnswerCallbackQuery, GetChat, GetMe, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from saqbol.bot.logic import SaqBolCore
from saqbol.bot.main import build_dispatcher
from saqbol.config import Settings
from saqbol.storage.db import Storage

_ids = itertools.count(1)


class RecordingSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.sent: list[SendMessage] = []

    async def make_request(self, bot, method, timeout=None):
        if isinstance(method, SendMessage):
            self.sent.append(method)
            return Message(message_id=next(_ids), date=dt.datetime.now(),
                           chat=Chat(id=method.chat_id, type="private"), text=method.text)
        if isinstance(method, GetMe):
            return User(id=999, is_bot=True, first_name="SaqBol", username="saqbol_demo_bot")
        if isinstance(method, AnswerCallbackQuery):
            return True
        if isinstance(method, GetChat):
            raise RuntimeError("not needed in tests")
        raise NotImplementedError(type(method).__name__)

    async def stream_content(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError

    async def close(self):
        pass


def user(uid, name):
    return User(id=uid, is_bot=False, first_name=name)


def msg_update(uid, name, text):
    m = Message(message_id=next(_ids), date=dt.datetime.now(), chat=Chat(id=uid, type="private", first_name=name),
                from_user=user(uid, name), text=text)
    return Update(update_id=next(_ids), message=m)


def cb_update(uid, name, data):
    m = Message(message_id=next(_ids), date=dt.datetime.now(), chat=Chat(id=uid, type="private", first_name=name),
                text="...")
    cq = CallbackQuery(id=str(next(_ids)), from_user=user(uid, name), chat_instance="x", data=data, message=m)
    return Update(update_id=next(_ids), callback_query=cq)


_DP = {}


@pytest.fixture
def env(analyzer):
    session = RecordingSession()
    bot = Bot(token="123456:TEST_TOKEN_FOR_UNIT_TESTS_ONLY_xx", session=session)
    core = SaqBolCore(Storage(":memory:"), analyzer, Settings(hash_secret="t", llm_enabled=False))
    # router модуля подключается к диспетчеру один раз; для каждого теста подменяем только core
    if "dp" not in _DP:
        _DP["dp"] = build_dispatcher(core)
    dp = _DP["dp"]
    dp["core"] = core

    def feed(update):
        session.sent.clear()
        asyncio.run(dp.feed_update(bot, update))
        return [(m.chat_id, m.text) for m in session.sent]

    return feed, core


def test_full_flow_language_check_link_notify_delete(env):
    feed, core = env
    elder, rel = (11, "Бабушка"), (22, "Внучка")

    out = feed(msg_update(*elder, "/start"))
    assert "Выберите язык" in out[0][1]
    feed(cb_update(*elder, "lang:kk"))
    assert core.db.get_language(elder[0]) == "kk"
    feed(cb_update(*elder, "lang:ru"))

    out = feed(msg_update(*elder, "Ваш код входа: 4821. Никому его не сообщайте."))
    assert out[0][1].startswith("ℹ️ Низкий риск.")

    # пожилой пользователь даёт согласие и получает одноразовую ссылку
    out = feed(cb_update(*elder, "link:allow"))
    link_text = out[0][1]
    assert "https://t.me/saqbol_demo_bot?start=link_" in link_text
    token = link_text.split("start=link_")[1].split()[0]

    # родственник открывает ссылку и соглашается
    out = feed(msg_update(*rel, f"/start link_{token}"))
    assert "приглашает" in out[0][1]
    out = feed(cb_update(*rel, f"link:accept:{token}"))
    chats = [c for c, _ in out]
    assert elder[0] in chats  # пожилому пользователю пришёл запрос на подтверждение
    link_id = core.db._conn.execute("SELECT link_id FROM links").fetchone()[0]

    high = "Служба безопасности банка: продиктуйте код из SMS, иначе карта будет заблокирована. Срочно."
    out = feed(msg_update(*elder, high))
    assert all(c != rel[0] for c, _ in out), "до подтверждения пожилым пользователем уведомлений нет"

    feed(cb_update(*elder, f"link:confirm:{link_id}"))
    out = feed(msg_update(*elder, high + " Повторяю."))
    to_rel = [t for c, t in out if c == rel[0]]
    assert len(to_rel) == 1 and "высокий риск" in to_rel[0]
    assert "продиктуйте" not in to_rel[0].lower()

    out = feed(msg_update(*elder, high + " Повторяю."))
    assert all(c != rel[0] for c, _ in out), "повторное уведомление об одном запросе не отправляется"

    out = feed(cb_update(*elder, "del:yes"))
    assert "Данные удалены" in out[0][1]
    assert core.db.active_relatives(elder[0]) == []


def test_unsupported_and_menu_buttons(env):
    feed, core = env
    m = msg_update(5, "U", "x")
    sticker_msg = m.message.model_copy(update={"text": None})
    out = feed(Update(update_id=next(_ids), message=sticker_msg))
    assert "только текст" in out[0][1]
    out = feed(msg_update(5, "U", "Проверить сообщение"))
    assert "Пришлите текст" in out[0][1]
    out = feed(msg_update(5, "U", "Связаться с близким"))
    assert "нет подключённого" in out[0][1]


def test_voice_disabled_by_default(env):
    from aiogram.types import Voice

    feed, _ = env
    m = msg_update(6, "V", "x").message.model_copy(
        update={"text": None, "voice": Voice(file_id="f", file_unique_id="u", duration=3)})
    out = feed(Update(update_id=next(_ids), message=m))
    assert "Голосовые сообщения пока не включены" in out[0][1]
