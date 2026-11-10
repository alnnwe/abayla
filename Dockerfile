# Образ бота с baseline-моделью (без torch, чтобы образ был лёгким).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY abayla ./abayla
COPY config ./config
COPY data/demo ./data/demo
# Модель обучается при сборке, чтобы образ был самодостаточным и воспроизводимым (seed=42)
RUN python -m abayla.training.train --models tfidf_wordchar,rules
RUN useradd --create-home abayla && mkdir -p /app/state && chown -R abayla /app/state
USER abayla
ENV ABAYLA_DB_PATH=/app/state/abayla.sqlite3
CMD ["python", "-m", "abayla.bot.main"]
