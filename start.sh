#!/bin/sh
python /app/main.py  # Run ML pipeline first
uvicorn app:app --host 0.0.0.0 --port 5000  # Start FastAPI after ML pipeline completes
