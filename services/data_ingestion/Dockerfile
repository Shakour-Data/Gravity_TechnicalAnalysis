FROM python:3.12-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

COPY services/data_ingestion/requirements.txt /tmp/ingestion-requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && pip install --upgrade pip \
    && pip install -r /tmp/ingestion-requirements.txt \
    && rm -rf /var/lib/apt/lists/*

COPY services/data_ingestion /app

ENV PYTHONPATH=/app

CMD ["python", "main.py", "load-all-prices"]
