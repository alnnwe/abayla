# Abayla — AI-помощник против мошенничества для пожилых пользователей (RU/KZ)

Название: каз. «абайла» — «будь осторожен».

Пользователь пересылает сообщение, ссылку или голосовой пересказ звонка. Abayla отвечает на русском или
казахском: **уровень риска** (низкий / не определён / высокий), **какие признаки** найдены и **одно действие**.
При высоком риске уведомляет близкого — только если оба заранее подтвердили подключение.

Решение принимает **собственный классификатор** (TF-IDF + логистическая регрессия по умолчанию; XLM-RoBERTa —
альтернатива). LLM (необязательно) только переформулирует объяснение.

> ⚠️ Модели обучены на **демонстрационном синтетическом наборе** (286 текстов). Бот показывает это в каждом ответе.
> Метрики в `reports/` — проверка конвейера, а не качество на реальных сообщениях. Цели precision ≥ 0.85 и
> recall ≥ 0.90 одновременно **не достигнуты** (см. `docs/report.md`).

## Быстрый старт (без Telegram)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m abayla.training.train                         # демо-данные → разбиение → 3 модели → пороги на val
python -m abayla.cli check "Продиктуйте код из SMS, иначе карту заблокируют"
python -m abayla.cli --lang kk check "Кіру кодыңыз: 4821. Оны ешкімге айтпаңыз."
python -m abayla.cli repl                               # интерактивная проверка
python -m abayla.cli scenarios                          # проверочные пары (уведомление с кодом / просьба кода и др.)
python -m pytest -q                                     # тесты
```

## Telegram-бот

1. Создайте бота у @BotFather, получите токен.
2. `cp .env.example .env`, заполните `BOT_TOKEN` и `ABAYLA_HASH_SECRET`.
3. `python -m abayla.bot.main`

Команды: `/start /help /language /settings /privacy`. Кнопки: «Проверить сообщение», «Подключить близкого»,
«Связаться с близким», «Бот ошибся», «Удалить мои данные». Бот работает только в личных чатах.

**Docker** (не проверялся в среде разработки — Docker там недоступен):
```bash
cp .env.example .env   # заполнить BOT_TOKEN и ABAYLA_HASH_SECRET
docker compose up --build
```
Образ содержит только baseline (без torch); модель обучается при сборке.

## Эксперименты

```bash
make eda                 # reports/eda/
make train               # baseline + вариант с аугментацией
pip install -r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
make train-xlmr          # XLM-R на CPU (~4–5 мин на 1 ядре, ~2.4 ГБ RAM), эмбеддинги заморожены
make evaluate            # reports/results.md, metrics.json, figures/, errors/
```
GPU-вариант: `notebooks/train_xlmr_colab.ipynb`. Переключение бота на трансформер без изменения кода:
`ABAYLA_MODEL_DIR=artifacts/models/xlmr python -m abayla.bot.main`.

## Необязательные модули

- **LLM-объяснения:** `LLM_ENABLED=1`, `ANTHROPIC_API_KEY=...`. Перед отправкой текст обезличивается; ответ
  проверяется (JSON, длина, язык, запрещённые фразы, номера, ссылки); при ошибке/тайм-ауте — шаблон.
- **Голос:** `pip install -r requirements-voice.txt`, `VOICE_ENABLED=1`. Бот показывает распознанный текст
  и даёт исправить его до проверки; аудиофайл удаляется сразу.

## Структура

```
abayla/
  config.py               настройки из переменных окружения
  data/                   schema (формат и валидация), demo_dataset, prepare (импорт CSV, разбиение по группам, аугментация), eda
  features/               signals (признаки RU/KZ с учётом отрицаний), url_analysis (без открытия страниц)
  models/                 base (общий интерфейс, пороги), rules, tfidf, transformer
  training/               train, train_transformer, thresholds (выбор на validation), env_info
  evaluation/evaluate.py  метрики, ДИ, графики, ошибки, задержка, learning curve
  service/                analyzer (модель + признаки + политика), privacy (обезличивание, фильтр логов)
  explain/                templates (RU/KZ), llm (необязательно), service
  storage/db.py           SQLite: язык, согласия, привязки, дедупликация уведомлений, отзывы
  bot/                    logic (без Telegram), handlers (aiogram 3), texts (RU/KZ), main
  voice/asr.py            faster-whisper, временные файлы
  cli.py                  локальная проверка без токена
tests/                    62 теста
docs/                     report, dataset_card, annotation_guide, responsible_ai, demo_script,
                          presentation_plan, team_contribution_template, pilot_study_template, criteria_coverage
config/url_rules.json     эвристики ссылок
data/demo/                демонстрационный набор
reports/                  EDA, метрики, графики, ошибочные предсказания
```

## Что требуется для реального запуска

- `BOT_TOKEN` (Telegram), `ABAYLA_HASH_SECRET`; при желании `ANTHROPIC_API_KEY`.
- Реальные размеченные данные (цель — 5 000 проверенных примеров) и повторная оценка — см. `docs/annotation_guide.md`.
- Пилот с пользователями — см. `docs/pilot_study_template.md`. Юридическая проверка обработки персональных данных.
