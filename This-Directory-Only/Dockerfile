FROM python:3.12-slim

WORKDIR /app

# System deps: gcc for psycopg/numpy compilation, libpq-dev for PostgreSQL driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY This-Directory-Only/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the split-layout files the backend uses at runtime
COPY This-Directory-Only/backend ./This-Directory-Only/backend
COPY This-Directory-Only/chatgpt ./This-Directory-Only/chatgpt
COPY This-Directory-Only/mathpix processor ./This-Directory-Only/mathpix processor
COPY This-Directory-Only/csv files of problems ./This-Directory-Only/csv files of problems
COPY Local_Deployed_Shared/glossary_to_csv.py ./Local_Deployed_Shared/glossary_to_csv.py
COPY Local_Deployed_Shared/split_LADR4e_chapters.py ./Local_Deployed_Shared/split_LADR4e_chapters.py

# Persistent volume mount point — Fly.io mounts /data here
RUN mkdir -p /data/storage /data/user_data

WORKDIR /app/This-Directory-Only/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
