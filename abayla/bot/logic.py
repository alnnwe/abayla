"""Логика бота без зависимости от Telegram. aiogram-обработчики только вызывают эти методы
и отправляют результат. Благодаря этому сценарии проверяются тестами без токена."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from abayla.bot.texts import t
from abayla.config import Settings, get_settings
from abayla.explain.service import explain
from abayla.service.analyzer import Analysis, Analyzer, EmptyTextError
from abayla.service.privacy import anonymize, text_fingerprint
from abayla.storage.db import LinkError, Storage

log = logging.getLogger("abayla.bot")


@dataclass
class Outgoing:
    chat_id: int
    text: str
    kind: str = "message"


@dataclass
class CheckResult:
    reply: str
    analysis: Analysis | None = None
    ask_clarification: bool = False
    outgoing: list[Outgoing] = field(default_factory=list)


class AbaylaCore:
    def __init__(self, storage: Storage, analyzer: Analyzer, settings: Settings | None = None, explain_fn=explain):
        self.db = storage
        self.analyzer = analyzer
        self.settings = settings or get_settings()
        self.explain = explain_fn

    def lang(self, user_id: int) -> str:
        return self.db.get_language(user_id)

    # ---------- проверка ----------
    def check(self, user_id: int, text: str | None, display_name: str = "", allow_clarification: bool = True) -> CheckResult:
        self.db.ensure_user(user_id)
        lang = self.lang(user_id)
        try:
            a = self.analyzer.analyze(text or "")
        except EmptyTextError:
            return CheckResult(reply=t(lang, "empty"))
        reply = self.explain(a, lang, original_text=text).render(lang)
        res = CheckResult(reply=reply, analysis=a)
        # логируем только метаданные: без текста, кодов и ссылок
        log.info("check level=%s model=%s signals=%d latency_ms=%s", a.level, a.model_name, len(a.signals), a.latency_ms)
        if a.level == "high":
            res.outgoing = self._notify_relatives(user_id, text or "", display_name)
            if res.outgoing:
                res.reply += "\n\n" + t(lang, "relative_notified")
        if a.needs_clarification and allow_clarification:
            res.ask_clarification = True
            res.reply += "\n\n" + t(lang, "clarify")
        return res

    def check_with_clarification(self, user_id: int, original: str, clarification: str, display_name: str = "") -> CheckResult:
        """Второй и последний шаг: больше одного уточняющего вопроса не задаём."""
        return self.check(user_id, f"{original}\n{clarification}", display_name, allow_clarification=False)

    def _notify_relatives(self, user_id: int, text: str, display_name: str) -> list[Outgoing]:
        user = self.db.get_user(user_id)
        if not user or not user.notify_enabled:
            return []
        relatives = self.db.active_relatives(user_id)  # только с подтверждением обеих сторон
        if not relatives:
            return []
        req_hash = text_fingerprint(self.settings.hash_secret, str(user_id), text)
        if not self.db.claim_notification(user_id, req_hash, self.settings.notify_dedup_hours,
                                          self.settings.notify_max_per_hour):
            log.info("notification suppressed (duplicate or rate limit)")
            return []
        name = display_name or "—"
        return [Outgoing(r, t(self.lang(r), "notify_relative", name=name), "notify") for r in relatives]

    # ---------- подключение близкого ----------
    def create_invite(self, elder_id: int, bot_username: str) -> str:
        """Вызывать только после нажатия «Да, разрешаю»."""
        token = self.db.create_link_token(elder_id, self.settings.link_ttl_minutes)
        url = f"https://t.me/{bot_username}?start=link_{token}"
        return t(self.lang(elder_id), "link_created", ttl=self.settings.link_ttl_minutes, url=url)

    def inspect_invite(self, token: str) -> tuple[int | None, str | None]:
        try:
            return self.db.peek_link_token(token), None
        except LinkError as e:
            return None, f"link_err_{e.args[0]}"

    def accept_invite(self, token: str, relative_id: int, relative_name: str) -> tuple[int | None, int | None, str | None]:
        """Возвращает (link_id, elder_id, ошибка)."""
        try:
            elder_id = self.db.peek_link_token(token)
            link_id = self.db.accept_link_token(token, relative_id, relative_name)
            return link_id, elder_id, None
        except LinkError as e:
            code = "self" if e.args[0] == "self_link" else e.args[0]
            return None, None, f"link_err_{code}"

    def elder_confirm(self, elder_id: int, link_id: int) -> int | None:
        if self.db.confirm_link_by_elder(elder_id, link_id):
            return int(self.db.get_link(link_id)["relative_id"])
        return None

    def contact_relative(self, user_id: int, display_name: str) -> CheckResult:
        lang = self.lang(user_id)
        relatives = self.db.active_relatives(user_id)
        if not relatives:
            return CheckResult(reply=t(lang, "no_relative"))
        out = [Outgoing(r, t(self.lang(r), "contact_request", name=display_name or "—"), "contact") for r in relatives]
        return CheckResult(reply=t(lang, "contact_sent"), outgoing=out)

    # ---------- обратная связь и данные ----------
    def feedback(self, user_id: int, kind: str, level: str | None, model_name: str | None,
                 last_text: str | None = None) -> str:
        lang = self.lang(user_id)
        self.db.add_feedback(user_id, kind, level, model_name)
        if kind != "wrong":
            return t(lang, "feedback_ok")
        msg = t(lang, "feedback_thanks")
        if last_text and self.db.add_shared_example(user_id, anonymize(last_text), level or "", "bot_wrong"):
            msg += "\n" + t(lang, "feedback_shared")
        return msg

    def settings_text(self, user_id: int) -> str:
        u = self.db.ensure_user(user_id)
        lang = self.lang(user_id)
        yes, no = t(lang, "on"), t(lang, "off")
        return t(lang, "settings", lang="Русский" if lang == "ru" else "Қазақша",
                 notify=yes if u.notify_enabled else no, share=yes if u.share_examples else no,
                 n=len(self.db.active_relatives(user_id)))

    def delete_all(self, user_id: int) -> str:
        lang = self.lang(user_id)
        self.db.delete_user_data(user_id)
        return t(lang, "deleted")
