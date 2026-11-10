import numpy as np
import pytest

from abayla.features.signals import detect_signals
from abayla.features.url_analysis import analyze_url, extract_urls
from abayla.models.base import load_model
from abayla.service.analyzer import EmptyTextError

NOTIFICATION_VS_REQUEST = [
    ("Ваш код входа: 4821. Никому его не сообщайте.", "Продиктуйте код из SMS."),
    ("Код подтверждения: 739104. Если вы не совершали операцию, не сообщайте код.", "Сообщите код подтверждения, который придёт."),
    ("Кіру кодыңыз: 4821. Оны ешкімге айтпаңыз.", "SMS-тегі кодты айтыңыз."),
    ("Ваш одноразовый пароль: 0937. Сотрудники банка никогда не спрашивают пароли.", "Назовите пароль от мобильного банка."),
]


@pytest.mark.parametrize("notification, code_req", NOTIFICATION_VS_REQUEST)
def test_code_notification_vs_code_request_signals(notification, code_req):
    n = detect_signals(notification).signals
    r = detect_signals(code_req).signals
    assert "code_request" not in n and "code_notification" in n
    assert "code_request" in r and "code_notification" not in r


@pytest.mark.parametrize("notification, code_req", NOTIFICATION_VS_REQUEST)
def test_code_pair_levels_differ(analyzer, notification, code_req):
    assert analyzer.analyze(notification).level == "low"
    assert analyzer.analyze(code_req).level != "low"


def test_secrecy_not_triggered_by_code_warning():
    assert "secrecy" not in detect_signals("Код для регистрации: 2281. Никому не сообщайте этот код.").signals
    assert "secrecy" in detect_signals("Никому не рассказывайте о нашем разговоре.").signals


def test_antifraud_warning_not_safe_account():
    s = detect_signals("Банк никогда не предлагает перевести деньги на безопасный счёт.").signals
    assert "safe_account" not in s


def test_url_extraction_and_flags():
    urls = extract_urls("Войдите: http://demobank-verify.example.com/login, а также www.example.org/page.")
    assert urls == ["http://demobank-verify.example.com/login", "www.example.org/page"]
    r = analyze_url("http://203.0.113.5/payout")
    assert "ip_host" in r.flags and r.suspicious
    r = analyze_url("https://pension-bonus.example.org@203.0.113.7/reg")
    assert "userinfo_at" in r.flags
    r = analyze_url("http://demobank-verify.example.com/login")
    assert "brand_lookalike" in r.flags
    assert not analyze_url("https://www.example.com/orders").suspicious


def test_empty_text_raises(analyzer):
    with pytest.raises(EmptyTextError):
        analyzer.analyze("   ")


def test_long_text_truncated(analyzer):
    a = analyzer.analyze("Обычное сообщение. " * 2000)
    assert a.truncated
    assert a.level in {"low", "uncertain", "high"}


def test_short_text_with_signal_asks_clarification(analyzer):
    a = analyzer.analyze("скинь 5000")
    assert a.level == "uncertain" and a.needs_clarification


def test_model_save_load_roundtrip(trained_dir, tmp_path):
    m = load_model(trained_dir / "tfidf_wordchar")
    texts = ["Продиктуйте код", "Мам, я доехал"]
    s1 = m.predict_scores(texts)
    m.save(tmp_path / "copy")
    m2 = load_model(tmp_path / "copy")
    assert np.allclose(s1, m2.predict_scores(texts))
    assert m2.meta.thresholds.low <= m2.meta.thresholds.binary <= m2.meta.thresholds.high
    assert m2.meta.demo_data is True
    assert m2.meta.seed == 42 and m2.meta.train_time_s is not None and m2.meta.hardware


def test_missing_model_dir_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="обучите"):
        load_model(tmp_path / "nope")
