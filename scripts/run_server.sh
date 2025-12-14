#!/usr/bin/env bash
# Run the FastAPI server locally on the recommended host/port
export PYTHONPATH=src
uvicorn gravity_tech.main:app --host 127.0.0.1 --port 8002 --workers 1 --log-level info
