#!/usr/bin/env bash
set -euo pipefail

# Create a Procfile for common Python web entrypoints
# Order of detection mimics the original one-liner
# Usage: ./scripts/create-procfile.sh

ROOT="$(pwd)"
PROCFILE="$ROOT/Procfile"

echo "Detecting app entrypoint to create $PROCFILE..."

if [ -f "start_server.sh" ]; then
  chmod +x start_server.sh
  echo "web: ./start_server.sh" > "$PROCFILE"
elif [ -f "start.sh" ]; then
  chmod +x start.sh
  echo "web: ./start.sh" > "$PROCFILE"
elif [ -f "run.sh" ]; then
  chmod +x run.sh
  echo "web: ./run.sh" > "$PROCFILE"
elif [ -f "main.py" ]; then
  if grep -q "fastapi\|FastAPI" main.py 2>/dev/null; then
    echo "web: uvicorn main:app --host 0.0.0.0 --port 8080" > "$PROCFILE"
  elif grep -q "flask\|Flask" main.py 2>/dev/null; then
    echo "web: gunicorn main:app --bind 0.0.0.0:8080" > "$PROCFILE"
  else
    echo "web: python main.py" > "$PROCFILE"
  fi
elif [ -f "app.py" ]; then
  if grep -q "flask\|Flask" app.py 2>/dev/null; then
    echo "web: gunicorn app:app --bind 0.0.0.0:8080" > "$PROCFILE"
  else
    echo "web: python app.py" > "$PROCFILE"
  fi
elif [ -f "server.py" ]; then
  echo "web: python server.py" > "$PROCFILE"
elif [ -f "index.py" ]; then
  echo "web: python index.py" > "$PROCFILE"
elif [ -f "manage.py" ]; then
  echo "web: gunicorn config.wsgi --bind 0.0.0.0:8080" > "$PROCFILE"
else
  echo "web: python main.py" > "$PROCFILE"
fi

# Show the created Procfile
if [ -f "$PROCFILE" ]; then
  echo "--- $PROCFILE ---"
  cat "$PROCFILE"
  echo "(Done)"
else
  echo "Failed to create $PROCFILE" >&2
  exit 1
fi
