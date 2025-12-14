# PowerShell script to run the FastAPI server on Windows (recommended 127.0.0.1:8002)
$env:PYTHONPATH = 'src'
uvicorn gravity_tech.main:app --host 127.0.0.1 --port 8002 --workers 1 --log-level info
