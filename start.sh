#!/bin/bash

# Run the ML pipeline in the background (does not block execution)
python /app/main.py &

# Keep FastAPI running in the foreground (keeps the container alive)
exec uvicorn app:app --host 0.0.0.0 --port 5000
