# PDF Split Tool Backend

FastAPI backend that wraps the existing PDF processing scripts and persists jobs in PostgreSQL.

## Quick start

1) Create and activate a virtualenv.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Set environment variables (copy `.env.example`).

4) Initialize the database:

```bash
python scripts/init_db.py
```

5) Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment

- `DATABASE_URL` supports local PostgreSQL and Supabase's connection string.
- `STORAGE_DIR` is where uploaded PDFs and split chapters are stored.
- `OPENAI_API_KEY` and Mathpix credentials are only required for `auto_toc=true` jobs.

## API summary

- `POST /auth/signup` -> create account, returns JWT.
- `POST /auth/login` -> returns JWT.
- `POST /jobs` -> upload PDF + optional chapters CSV.
- `GET /jobs/{job_id}` -> job status.
- `GET /jobs/{job_id}/chapters` -> list chapter files.
- `GET /jobs/{job_id}/chapters/{chapter_id}/download` -> download a split PDF.

## Notes

- If you pass `auto_toc=true`, the backend runs `glossary_to_csv.py` to extract a TOC and build `toc_chapters.csv`.
- If you already have a chapters CSV, upload it with the PDF and set `auto_toc=false`.
