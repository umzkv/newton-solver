FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Порт по умолчанию
EXPOSE 5050
ENV PORT=5050

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5050} newton_web:app"]
