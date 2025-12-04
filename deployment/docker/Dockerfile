FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential libffi-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY pyproject.toml /app/
COPY README.md /app/

# Install package in editable mode
RUN pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "gravity_tech.main:app", "--host", "0.0.0.0", "--port", "8000"]
