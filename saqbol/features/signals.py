"""Обнаружение интерпретируемых признаков мошенничества (RU/KK).

Эти признаки используются в трёх местах:
1) в системе правил (модель №1 для сравнения);
2) в объяснениях — объясняются ТОЛЬКО реально найденные признаки;
3) в страховочной политике сервиса (сильный признак не даёт ответить «низкий риск»).

Важная деталь — отрицания: «Никому не сообщайте код» (служебное уведомление) не равно
«Продиктуйте код» (просьба передать код).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from saqbol.features.url_analysis import analyze_text_urls


def normalize(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def clauses(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[.!?;\n]+", text) if c.strip()]


# --- отрицание/предупреждение внутри фрагмента
NEG_CLAUSE = re.compile(
    r"(\b(никогда|ни в коем случае)\b|\bне\s+(\w+\s+){0,2}(просят|предлагают|переводит|спрашивают|"
    r"сообщайте|называйте|говорите|передавайте|вносите|диктуйте|отправляйте|вводите)\b|"
    r"сұрамайды|ұсынбайды|айтпаңыз|бермеңіз|жасамаңыз|енгізбеңіз|ешқашан)"
)


def _negated_before(clause: str, start: int) -> bool:
    window = clause[max(0, start - 16): start]
    return bool(re.search(r"(\bне\s*$|\bне\s+\w+\s*$|никому\s*(его|их|это)?\s*$|никогда\s*$)", window))


CODE_OBJ = r"(код\w*|парол\w*|pin|пин\b|cvv|cvc|кодов\w+|құпиясөз\w*|кодты|кодтық)"
REQ_VERB_RU = (
    r"(продикт\w*|назов(и|ите)\b|назвать|сообщ(и|ите)\b|скаж(и|ите)\b|отправ(ь|ьте)\b|"
    r"перешл(и|ите)\b|пришл(и|ите)\b|подтверд\w*\s+(\w+\s+){0,3}кодом|нужно\s+(назвать|сказать|сообщить))"
)
REQ_VERB_KK = r"(айтыңыз\w*|айтып беріңіз|айтуыңыз керек|айтшы|жіберіңіз|атаңыз|хабарлаңыз|кодпен\s+(\w+\s+){0,2}растаңыз)"
ENTER_VERB = r"(введите|ввести|енгізіңіз|енгізу керек)"


def _code_request(norm: str, has_url: bool) -> bool:
    for cl in clauses(norm):
        if not re.search(CODE_OBJ, cl):
            continue
        for verb_re in (REQ_VERB_RU, REQ_VERB_KK) + ((ENTER_VERB,) if has_url else ()):
            for m in re.finditer(verb_re, cl):
                if not _negated_before(cl, m.start()):
                    return True
    return False


CODE_NOTIFICATION = re.compile(
    r"((ваш|ваше)\s+(\w+\s+){0,2}(код|парол)\w*|код\w*\s+(\w+\s+){0,3}\S*:\s*\d{3,8}|"
    r"кодыңыз|коды:\s*\d|құпиясөзіңіз|бір реттік)"
)
WARNING_PHRASE = re.compile(
    r"(никому|не сообщайте|не передавайте|не называйте|никогда не|ешкімге|айтпаңыз|бермеңіз|ешқашан|"
    r"если это были не вы|егер бұл сіз болмасаңыз)"
)


@dataclass(frozen=True)
class SignalDef:
    id: str
    weight: float
    strong: bool = False


SIGNALS: dict[str, SignalDef] = {
    s.id: s
    for s in [
        SignalDef("code_request", 3.0, strong=True),
        SignalDef("card_data_request", 3.0, strong=True),
        SignalDef("safe_account", 3.0, strong=True),
        SignalDef("remote_access", 2.5, strong=True),
        SignalDef("personal_data_request", 2.0, strong=True),
        SignalDef("guaranteed_income", 2.0),
        SignalDef("suspicious_link", 2.0),
        SignalDef("money_transfer_request", 1.5),
        SignalDef("authority_impersonation", 1.0),
        SignalDef("secrecy", 1.5),
        SignalDef("relative_new_number", 1.5),
        SignalDef("relative_trouble", 1.5),
        SignalDef("prize_fee", 1.5),
        SignalDef("prepayment", 1.5),
        SignalDef("urgency", 1.0),
        SignalDef("threat", 1.0),
        SignalDef("link_present", 0.3),
        # «безопасный» признак — уменьшает оценку правил
        SignalDef("code_notification", -2.5),
    ]
}

PATTERNS: dict[str, list[str]] = {
    "card_data_request": [
        r"(назов|сообщ(и|ите)|продикт|введите|укажите|отправьте)\w*[^.]{0,40}"
        r"(номер\w*\s+карт|данные\s+карт|cvv|cvc|три\s+цифры|срок\w*\s+действия)",
        r"(укажите|введите)\s+(номер\s+)?карт\w*",
        r"(номер\w*\s+карт\w*|данные\s+карт\w*|cvv|cvc)[^.]{0,40}(назов|сообщ(и|ите)|продикт|введите|укажите)",
        r"(карта\s+нөмір\w*|жарамдылық мерзім\w*|үш сан\w*|cvv|карта деректер\w*)[^.]{0,60}(айтыңыз|енгізіңіз|көрсетіңіз)",
    ],
    "safe_account": [
        r"(безопасн\w+|защищенн\w+|резервн\w+)\s+сч[её]т\w*",
        r"(қауіпсіз|қорғалған|резервтік)\s+шот\w*",
    ],
    "remote_access": [
        r"(удаленн\w+\s+(доступ|поддержк)\w*|демонстраци\w+\s+экрана|программ\w+\s+удаленного|\.apk\b|"
        r"скачайте\s+(программу|приложение|новую версию)|установите\s+(приложение|программу))",
        r"(қашықтан\s+(қолдау|қол жеткізу)|экранды көрсету)",
    ],
    "personal_data_request": [
        r"(назов|сообщ|продикт|укажите|отправьте)\w*[^.]{0,40}(иин|удостоверени\w+|паспорт\w*|кодов\w+\s+слов\w*)",
        r"(жсн\w*|куәлік нөмір\w*|кодтық сөз\w*)[^.]{0,40}(айтыңыз|жіберіңіз)",
    ],
    "guaranteed_income": [
        r"(гарантир\w+\s+(доход|доходност|прибыл)\w*|гаранти\w+\s+прибыли|без риска|удвои\w*|"
        r"\d+\s*%\s*(в|каждую)\s*(день|неделю)|пассивн\w+\s+доход\w*|доход\w*\s+от\s+\d)",
        r"(кепілді табыс|тәуекелсіз|екі есе көбейт\w*|күніне\s+\d+\s*%|аптасына\s+\d+\s*%|"
        r"\d+\s*%\s*кепілдік|пассивті табыс|табыс табыңыз)",
    ],
    "money_transfer_request": [
        r"\b(переведи\w*|перевести|скинь\w*|кинь|оплатите|внесите|снимите\s+наличные|передать\s+\d|"
        r"оформите\s+(\w+\s+)?кредит|занять\s+\d|займи)\b",
        r"(аударыңыз|аударшы|аудару керек|салып жіберші|салыңыз|төлеңіз|төлеу керек|қарызға бере|беру керек|ақша\s+(тез\s+)?керек)",
    ],
    "authority_impersonation": [
        r"(служб\w+\s+безопасност\w+|следовател\w+|полици\w+|прокур\w+|\bкнб\b|национальн\w+\s+банк\w*|"
        r"сотрудник\w*\s+(банка|полиции)|специалист\w*\s+банка|пенсионн\w+\s+фонд\w*|налогов\w+|"
        r"оператор\w*\s+сотов\w+|департамент\w*|дежурн\w+)",
        r"(қауіпсіздік қызмет\w*|тергеуші\w*|полиция\w*|прокурор|ұқк|ұлттық банк|банк қызметкер\w*|"
        r"банк маман\w*|зейнетақы қоры|салық қызмет\w*|ұялы байланыс)",
    ],
    "secrecy": [
        r"(никому не (говори|рассказыва|сообща)\w*\s+(о|об|про)\b|не говори\w*\s+(маме|папе|родственник\w*|близким)|"
        r"(маме|папе)\s+не говори|не звони\w*\s+(маме|на мой номер)|только никому не говори)",
        r"(әңгіме туралы ешкімге|бұл туралы ешкімге|анама\s+(қоңырау шалма|айтпа)|әкеме айтпа|"
        r"ешкімге айтпа\b|нөміріме қоңырау шалма)",
    ],
    "relative_new_number": [
        r"(нов\w+\s+номер|с чужого|с телефона\s+(подруги|друга)|телефон\s+разбил\w*|старый\s+потерял|"
        r"пишу\s+с\s+(нового|чужого))",
        r"(жаңа нөмір|бөтен телефон|телефоным сын\w*|құрбымның телефонынан|досымның телефонынан|"
        r"жаңасынан жазып|ескі нөмірімді жоғалт\w*)",
    ],
    "relative_trouble": [
        r"(попал\w*\s+в\s+(аварию|дтп|неприятност\w+|беду)|задержан\w*|в больнице|в реанимации|ограбили|"
        r"сбил\w*\s+человека|влип\w*|после дтп|на операцию)",
        r"(апатқа түст\w*|ұсталды|ауруханада\w*|реанимация\w*|тонап кетті|қағып кетті|қиын жағдайға|отаға)",
    ],
    "prepayment": [
        r"(предоплат\w*|залог\w*|после\s+(оплаты|перевода)|растаможк\w*|бронь после|"
        r"оплатите\s+(доставку|курьеру|хранение)|оплатить\s+курьеру)",
        r"(алдын ала төлем|кепіл ақша\w*|төлегеннен кейін|аударғаннан кейін|жеткізу ақысы\w*)",
    ],
    "urgency": [
        r"(срочно|немедленно|сегодня же|в течение\s+\d+\s+час\w*|в течение часа|через час|только сегодня|"
        r"прямо сейчас|как можно скорее|до вечера)",
        r"(тез арада|шұғыл|дереу|\d+\s+сағат ішінде|бір сағат|\bтез\b)",
    ],
    "threat": [
        r"(заблокир\w+|арест\w*|уголовн\w+\s+дел\w*|штраф\w*|отключ(им|ат)\b|будет\s+удален|будет\s+закрыт|"
        r"приостановлен\w*|перестанет работать)",
        r"(бұғатта\w*|тыйым салын\w*|қылмыстық іс|айыппұл|өшіреміз|жойылады|жабылады|тоқтатылады|жұмысын тоқтатады)",
    ],
}
PRIZE = re.compile(
    r"(выиграл\w*|победител\w*|приз\w*|выигрыш\w*|компенсаци\w+|выплат\w+|доплат\w+|бонус\w*|подарк\w*|"
    r"ұтып ал\w*|ұтыс\w*|жеңімпаз\w*|сыйлық\w*|өтемақы\w*|төлем есептелді|бонус)"
)
FEE = re.compile(
    r"(оплатите|комисси\w+|\bсбор\w*|налог\w*|пошлин\w*|оплатите доставк\w*|төлеңіз|комиссия|салық төле\w*|\bбаж\b|ақысын төле\w*)"
)
NEGATABLE = {"safe_account", "prepayment", "card_data_request", "personal_data_request", "money_transfer_request"}


@dataclass
class SignalResult:
    signals: list[str]
    url_reports: list

    @property
    def strong(self) -> list[str]:
        return [s for s in self.signals if SIGNALS[s].strong]

    @property
    def risk_signals(self) -> list[str]:
        return [s for s in self.signals if SIGNALS[s].weight > 0 and s != "link_present"]


def _match_any(patterns: list[str], text: str, negatable: bool) -> bool:
    for cl in clauses(text):
        for p in patterns:
            if re.search(p, cl):
                if negatable and NEG_CLAUSE.search(cl):
                    continue
                return True
    return False


def detect_signals(text: str) -> SignalResult:
    norm = normalize(text)
    url_reports = analyze_text_urls(text or "")
    found: list[str] = []
    has_url = bool(url_reports)

    if _code_request(norm, has_url):
        found.append("code_request")
    for sid, pats in PATTERNS.items():
        if _match_any(pats, norm, negatable=sid in NEGATABLE):
            found.append(sid)
    if PRIZE.search(norm) and FEE.search(norm):
        found.append("prize_fee")
    if has_url:
        found.append("link_present")
        if any(r.suspicious for r in url_reports):
            found.append("suspicious_link")

    is_notification = (
        "code_request" not in found
        and CODE_NOTIFICATION.search(norm)
        and WARNING_PHRASE.search(norm)
    )
    if is_notification:
        found.append("code_notification")
        # «никому не сообщайте» в уведомлении о коде — не признак скрытности разговора
        if "secrecy" in found and not re.search(r"(разговор|әңгіме|маме|папе|анама|әкеме)", norm):
            found.remove("secrecy")

    order = list(SIGNALS)
    return SignalResult(signals=sorted(set(found), key=order.index), url_reports=url_reports)
