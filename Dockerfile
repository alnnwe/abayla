# Образ бота с baseline-моделью (без torch, чтобы образ был лёгким).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY saqbol ./saqbol
COPY config ./config
COPY data/demo ./data/demo
# Модель обучается при сборке, чтобы образ был самодостаточным и воспроизводимым (seed=42)
RUN python -m saqbol.training.train --models tfidf_wordchar,rules
RUN useradd --create-home saqbol && mkdir -p /app/state && chown -R saqbol /app/state
USER saqbol
ENV SAQBOL_DB_PATH=/app/state/saqbol.sqlite3
CMD ["python", "-m", "saqbol.bot.main"]
