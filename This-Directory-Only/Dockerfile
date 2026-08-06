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
# Lesson metadata for the first-encounter exposure guard (app/lessons.py):
# qmatrix_tags.json + kc_registry.json + lessons_structured.json. Lesson
# CONTENT is rendered by the static frontend; the backend only reads maps.
COPY Local_Deployed_Shared/lessons ./Local_Deployed_Shared/lessons
# The KC->atom crosswalk (app/kc_graph.py): the join between the ids the graph
# names concepts with and the ids BKT actually holds belief under. It is the
# only file the backend reads out of concept-graph/, and it is REQUIRED, not
# optional — missing, every concept reports the prior and the knowledge graph
# draws one flat colour that no amount of practice moves. /health publishes the
# row count so an image that lost it says so on the first request.
COPY Local_Deployed_Shared/concept-graph/kc_atom_crosswalk.json ./Local_Deployed_Shared/concept-graph/kc_atom_crosswalk.json
# numbers.npy (the visual-fixture image data) is vendored in backend/app/data
# — the ARENA content tree is gitignored and absent from this build context,
# so code_runner resolves the vendored copy first.

# Persistent volume mount point — Fly.io mounts /data here
RUN mkdir -p /data/storage /data/user_data

WORKDIR /app/This-Directory-Only/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
