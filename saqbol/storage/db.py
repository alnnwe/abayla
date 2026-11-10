"""SQLite-хранилище: язык, согласия, привязки родственников, дедупликация уведомлений, отзывы.

Минимизация данных: полные тексты проверяемых сообщений НЕ сохраняются. Исключение —
таблица shared_examples, куда попадает обезличенный текст только при отдельном согласии
пользователя (share_examples=1). Для дедупликации хранится HMAC-отпечаток, а не текст.
Одноразовые токены привязки хранятся в виде SHA-256.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    lang TEXT,
    notify_enabled INTEGER NOT NULL DEFAULT 1,
    share_examples INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS link_tokens (
    token_hash TEXT PRIMARY KEY,
    elder_id INTEGER NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    elder_id INTEGER NOT NULL,
    relative_id INTEGER NOT NULL,
    relative_name TEXT,
    elder_confirmed INTEGER NOT NULL DEFAULT 0,
    relative_confirmed INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    UNIQUE (elder_id, relative_id)
);
CREATE TABLE IF NOT EXISTS notifications (
    elder_id INTEGER NOT NULL,
    request_hash TEXT NOT NULL,
    sent_at REAL NOT NULL,
    PRIMARY KEY (elder_id, request_hash)
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    predicted_level TEXT,
    model_name TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS shared_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    text_anonymized TEXT NOT NULL,
    predicted_level TEXT,
    user_says TEXT,
    created_at REAL NOT NULL
);
"""


class LinkError(Exception):
    """Причина: not_found | expired | used | self_link"""


@dataclass
class UserSettings:
    user_id: int
    lang: str | None
    notify_enabled: bool
    share_examples: bool


class Storage:
    def __init__(self, path: str | Path = ":memory:", clock=time.time):
        self.path = str(path)
        self.clock = clock
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ---------- пользователи и настройки ----------
    def ensure_user(self, user_id: int) -> UserSettings:
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?)", (user_id, self.clock()))
        return self.get_user(user_id)

    def get_user(self, user_id: int) -> UserSettings | None:
        r = self._conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        if not r:
            return None
        return UserSettings(r["user_id"], r["lang"], bool(r["notify_enabled"]), bool(r["share_examples"]))

    def set_language(self, user_id: int, lang: str) -> None:
        if lang not in ("ru", "kk"):
            raise ValueError(lang)
        self.ensure_user(user_id)
        with self.tx() as c:
            c.execute("UPDATE users SET lang=? WHERE user_id=?", (lang, user_id))

    def get_language(self, user_id: int, default: str = "ru") -> str:
        u = self.get_user(user_id)
        return (u.lang if u and u.lang else default)

    def set_notify(self, user_id: int, enabled: bool) -> None:
        self.ensure_user(user_id)
        with self.tx() as c:
            c.execute("UPDATE users SET notify_enabled=? WHERE user_id=?", (int(enabled), user_id))

    def set_share_examples(self, user_id: int, enabled: bool) -> None:
        self.ensure_user(user_id)
        with self.tx() as c:
            c.execute("UPDATE users SET share_examples=? WHERE user_id=?", (int(enabled), user_id))

    # ---------- привязка родственника (согласие обеих сторон) ----------
    @staticmethod
    def _h(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create_link_token(self, elder_id: int, ttl_minutes: int = 30) -> str:
        """Вызывается ТОЛЬКО после того, как пожилой пользователь нажал «Разрешаю»."""
        self.ensure_user(elder_id)
        token = secrets.token_urlsafe(16)
        now = self.clock()
        with self.tx() as c:
            c.execute("UPDATE link_tokens SET used=1 WHERE elder_id=? AND used=0", (elder_id,))
            c.execute("INSERT INTO link_tokens VALUES (?, ?, ?, ?, 0)",
                      (self._h(token), elder_id, now, now + ttl_minutes * 60))
        return token

    def peek_link_token(self, token: str) -> int:
        """Проверяет токен без использования; возвращает elder_id."""
        r = self._conn.execute("SELECT * FROM link_tokens WHERE token_hash=?", (self._h(token),)).fetchone()
        if not r:
            raise LinkError("not_found")
        if r["used"]:
            raise LinkError("used")
        if r["expires_at"] < self.clock():
            raise LinkError("expired")
        return int(r["elder_id"])

    def accept_link_token(self, token: str, relative_id: int, relative_name: str = "") -> int:
        """Родственник подтвердил согласие. Токен сгорает. Связь ещё НЕ активна:
        нужно повторное подтверждение пожилого пользователя (защита от перехвата ссылки)."""
        elder_id = self.peek_link_token(token)
        if elder_id == relative_id:
            raise LinkError("self_link")
        self.ensure_user(relative_id)
        with self.tx() as c:
            c.execute("UPDATE link_tokens SET used=1 WHERE token_hash=?", (self._h(token),))
            c.execute(
                "INSERT INTO links(elder_id, relative_id, relative_name, elder_confirmed, relative_confirmed, active, created_at)"
                " VALUES (?, ?, ?, 0, 1, 0, ?) ON CONFLICT(elder_id, relative_id) DO UPDATE SET"
                " relative_confirmed=1, elder_confirmed=0, active=0, relative_name=excluded.relative_name",
                (elder_id, relative_id, relative_name[:64], self.clock()),
            )
            link_id = c.execute("SELECT link_id FROM links WHERE elder_id=? AND relative_id=?",
                                (elder_id, relative_id)).fetchone()[0]
        return int(link_id)

    def confirm_link_by_elder(self, elder_id: int, link_id: int) -> bool:
        with self.tx() as c:
            cur = c.execute(
                "UPDATE links SET elder_confirmed=1, active=1 WHERE link_id=? AND elder_id=? AND relative_confirmed=1",
                (link_id, elder_id),
            )
        return cur.rowcount == 1

    def reject_link(self, elder_id: int, link_id: int) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM links WHERE link_id=? AND elder_id=?", (link_id, elder_id))

    def get_link(self, link_id: int):
        return self._conn.execute("SELECT * FROM links WHERE link_id=?", (link_id,)).fetchone()

    def active_relatives(self, elder_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT relative_id FROM links WHERE elder_id=? AND active=1 AND elder_confirmed=1 AND relative_confirmed=1",
            (elder_id,),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def elders_of(self, relative_id: int) -> list[int]:
        rows = self._conn.execute("SELECT elder_id FROM links WHERE relative_id=? AND active=1", (relative_id,)).fetchall()
        return [int(r[0]) for r in rows]

    def unlink_all(self, user_id: int) -> int:
        with self.tx() as c:
            cur = c.execute("DELETE FROM links WHERE elder_id=? OR relative_id=?", (user_id, user_id))
        return cur.rowcount

    # ---------- уведомления: без повторов и с ограничением частоты ----------
    def claim_notification(self, elder_id: int, request_hash: str, dedup_hours: int = 24, max_per_hour: int = 3) -> bool:
        """True — уведомление можно отправить (и это зафиксировано). False — повтор или лимит."""
        now = self.clock()
        with self.tx() as c:
            c.execute("DELETE FROM notifications WHERE sent_at < ?", (now - dedup_hours * 3600,))
            if c.execute("SELECT 1 FROM notifications WHERE elder_id=? AND request_hash=?",
                         (elder_id, request_hash)).fetchone():
                return False
            recent = c.execute("SELECT COUNT(*) FROM notifications WHERE elder_id=? AND sent_at > ?",
                               (elder_id, now - 3600)).fetchone()[0]
            if recent >= max_per_hour:
                return False
            c.execute("INSERT INTO notifications VALUES (?, ?, ?)", (elder_id, request_hash, now))
        return True

    # ---------- обратная связь ----------
    def add_feedback(self, user_id: int, kind: str, predicted_level: str | None, model_name: str | None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO feedback(user_id, kind, predicted_level, model_name, created_at) VALUES (?,?,?,?,?)",
                      (user_id, kind, predicted_level, model_name, self.clock()))

    def add_shared_example(self, user_id: int, text_anonymized: str, predicted_level: str, user_says: str) -> bool:
        u = self.get_user(user_id)
        if not (u and u.share_examples):
            return False  # без отдельного согласия текст не сохраняем
        with self.tx() as c:
            c.execute("INSERT INTO shared_examples(user_id, text_anonymized, predicted_level, user_says, created_at)"
                      " VALUES (?,?,?,?,?)", (user_id, text_anonymized, predicted_level, user_says, self.clock()))
        return True

    # ---------- удаление ----------
    def delete_user_data(self, user_id: int) -> dict:
        counts = {}
        with self.tx() as c:
            for table, where in [
                ("links", "elder_id=? OR relative_id=?"),
                ("link_tokens", "elder_id=?"),
                ("notifications", "elder_id=?"),
                ("feedback", "user_id=?"),
                ("shared_examples", "user_id=?"),
                ("users", "user_id=?"),
            ]:
                params = (user_id,) * where.count("?")
                counts[table] = c.execute(f"DELETE FROM {table} WHERE {where}", params).rowcount
        return counts

    def stored_summary(self, user_id: int) -> dict:
        q = lambda sql, *p: self._conn.execute(sql, p).fetchone()[0]  # noqa: E731
        return {
            "settings": self.get_user(user_id) is not None,
            "links": q("SELECT COUNT(*) FROM links WHERE elder_id=? OR relative_id=?", user_id, user_id),
            "feedback": q("SELECT COUNT(*) FROM feedback WHERE user_id=?", user_id),
            "shared_examples": q("SELECT COUNT(*) FROM shared_examples WHERE user_id=?", user_id),
        }
