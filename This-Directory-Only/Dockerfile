FROM python:3.12-slim

WORKDIR /app

# System deps: gcc for psycopg/numpy compilation, libpq-dev for PostgreSQL driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies. torch MUST come from the cpu index first —
# plain PyPI serves the CUDA-bundled wheel (multi-GB, OOMs the VM); the cpu
# wheel is ~200MB and imports in ~1s. requirements.txt's torch>=2.0 is then
# already satisfied and skipped.
COPY This-Directory-Only/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the split-layout files the backend uses at runtime
COPY This-Directory-Only/backend ./This-Directory-Only/backend
COPY This-Directory-Only/chatgpt ./This-Directory-Only/chatgpt
COPY ["This-Directory-Only/mathpix processor", "./This-Directory-Only/mathpix processor"]
COPY ["This-Directory-Only/csv files of problems", "./This-Directory-Only/csv files of problems"]
COPY Local_Deployed_Shared/chatgpt ./Local_Deployed_Shared/chatgpt
COPY Local_Deployed_Shared/tools ./Local_Deployed_Shared/tools
# The grading harness redirects np.load('/delta_numbers.npy') to this file
# (code_runner.ARENA_NUMBERS_PATH) — without it every canonical visual/einops
# image fixture fails with FileNotFoundError on the backend (405KB).
COPY Local_Deployed_Shared/content/ARENA_3.0-main/chapter0_fundamentals/exercises/part0_prereqs/numbers.npy ./Local_Deployed_Shared/content/ARENA_3.0-main/chapter0_fundamentals/exercises/part0_prereqs/numbers.npy

# Persistent volume mount point — Fly.io mounts /data here
RUN mkdir -p /data/storage /data/user_data

WORKDIR /app/This-Directory-Only/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
