import pytest

from abayla.bot.logic import AbaylaCore
from abayla.config import Settings
from abayla.storage.db import LinkError, Storage

HIGH = "Здравствуйте, служба безопасности банка. Продиктуйте код из SMS, иначе счёт заблокируют. Срочно."
ELDER, RELATIVE, STRANGER = 1001, 2002, 3003


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def core(analyzer, clock):
    s = Settings(hash_secret="test", link_ttl_minutes=30, llm_enabled=False)
    return AbaylaCore(Storage(":memory:", clock=clock), analyzer, s)


def link(core):
    token = core.db.create_link_token(ELDER, 30)
    link_id, elder_id, err = core.accept_invite(token, RELATIVE, "Айгерим")
    assert err is None and elder_id == ELDER
    return link_id


def test_language_persists(tmp_path):
    db_path = tmp_path / "x.sqlite3"
    Storage(db_path).set_language(5, "kk")
    assert Storage(db_path).get_language(5) == "kk"  # новое соединение — значение сохранилось
    with pytest.raises(ValueError):
        Storage(db_path).set_language(5, "en")


def test_reply_uses_selected_language(core):
    core.db.set_language(ELDER, "kk")
    assert "Қауіп" in core.check(ELDER, HIGH).reply


def test_no_notification_without_link(core):
    res = core.check(ELDER, HIGH, "Бабушка")
    assert res.analysis.level == "high"
    assert res.outgoing == []


def test_no_notification_until_elder_confirms(core):
    link_id = link(core)
    assert core.db.active_relatives(ELDER) == []  # согласен только родственник
    assert core.check(ELDER, HIGH).outgoing == []
    assert core.elder_confirm(ELDER, link_id) == RELATIVE
    res = core.check(ELDER, HIGH, "Бабушка")
    assert [o.chat_id for o in res.outgoing] == [RELATIVE]
    assert HIGH not in res.outgoing[0].text  # текст переписки не пересылается
    assert "Продиктуйте" not in res.outgoing[0].text


def test_stranger_cannot_confirm_link(core):
    link_id = link(core)
    assert core.elder_confirm(STRANGER, link_id) is None
    assert core.db.active_relatives(ELDER) == []


def test_no_duplicate_notification(core):
    core.elder_confirm(ELDER, link(core))
    assert len(core.check(ELDER, HIGH).outgoing) == 1
    assert core.check(ELDER, HIGH).outgoing == []  # тот же запрос — без повтора
    assert core.check(ELDER, HIGH + "  ").outgoing == []  # отличия в пробелах не обходят защиту


def test_notification_rate_limit(core):
    core.elder_confirm(ELDER, link(core))
    sent = sum(len(core.check(ELDER, f"{HIGH} Вариант {i}.").outgoing) for i in range(6))
    assert sent == core.settings.notify_max_per_hour


def test_notifications_can_be_disabled(core):
    core.elder_confirm(ELDER, link(core))
    core.db.set_notify(ELDER, False)
    assert core.check(ELDER, HIGH).outgoing == []


def test_token_one_time_and_expiry(core, clock):
    token = core.db.create_link_token(ELDER, 30)
    core.accept_invite(token, RELATIVE, "A")
    assert core.accept_invite(token, STRANGER, "B")[2] == "link_err_used"
    token2 = core.db.create_link_token(ELDER, 30)
    clock.now += 31 * 60
    assert core.accept_invite(token2, STRANGER, "B")[2] == "link_err_expired"
    assert core.accept_invite("garbage", STRANGER, "B")[2] == "link_err_not_found"
    token3 = core.db.create_link_token(ELDER, 30)
    assert core.accept_invite(token3, ELDER, "self")[2] == "link_err_self"


def test_new_token_invalidates_previous(core):
    t1 = core.db.create_link_token(ELDER, 30)
    core.db.create_link_token(ELDER, 30)
    with pytest.raises(LinkError):
        core.db.peek_link_token(t1)


def test_tokens_stored_hashed(core):
    token = core.db.create_link_token(ELDER, 30)
    rows = core.db._conn.execute("SELECT token_hash FROM link_tokens").fetchall()
    assert all(token not in r[0] for r in rows)


def test_message_text_not_stored_without_consent(core):
    core.check(ELDER, HIGH)
    core.feedback(ELDER, "wrong", "high", "m", last_text=HIGH)
    dump = "\n".join(core.db._conn.iterdump())
    assert "Продиктуйте" not in dump


def test_shared_example_anonymized_with_consent(core):
    core.db.ensure_user(ELDER)
    core.db.set_share_examples(ELDER, True)
    msg = core.feedback(ELDER, "wrong", "high", "m", last_text="Мой ИИН 900101300123, код 4821, звоните +7 701 123 45 67")
    assert "обезличенный" in msg.lower()
    stored = core.db._conn.execute("SELECT text_anonymized FROM shared_examples").fetchone()[0]
    assert "900101300123" not in stored and "4821" not in stored and "701" not in stored


def test_delete_user_data(core):
    core.db.set_language(ELDER, "ru")
    core.elder_confirm(ELDER, link(core))
    core.check(ELDER, HIGH)
    core.feedback(ELDER, "wrong", "high", "m")
    core.delete_all(ELDER)
    s = core.db.stored_summary(ELDER)
    assert s == {"settings": False, "links": 0, "feedback": 0, "shared_examples": 0}
    assert core.db._conn.execute("SELECT COUNT(*) FROM notifications WHERE elder_id=?", (ELDER,)).fetchone()[0] == 0
    assert core.db.elders_of(RELATIVE) == []  # связь исчезла и у родственника


def test_contact_relative_requires_link(core):
    assert core.contact_relative(ELDER, "Бабушка").outgoing == []
    core.elder_confirm(ELDER, link(core))
    assert [o.chat_id for o in core.contact_relative(ELDER, "Бабушка").outgoing] == [RELATIVE]


def test_one_clarifying_question_only(core):
    first = core.check(ELDER, "скинь 5000")
    assert first.ask_clarification
    second = core.check_with_clarification(ELDER, "скинь 5000", "написал внук с нового номера")
    assert not second.ask_clarification


def test_empty_text(core):
    assert "пустое" in core.check(ELDER, "   ").reply
