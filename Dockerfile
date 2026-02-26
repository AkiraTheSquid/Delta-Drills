FROM python:3.12-slim

WORKDIR /app

# System deps: gcc for psycopg/numpy compilation, libpq-dev for PostgreSQL driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all whitelisted files (see .dockerignore)
COPY . .

# Persistent volume mount point — Fly.io mounts /data here
RUN mkdir -p /data/storage /data/user_data

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
