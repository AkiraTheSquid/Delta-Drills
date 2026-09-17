# PDF Split Tool Backend

FastAPI backend that wraps the existing PDF processing scripts and persists jobs in PostgreSQL.

## Quick start

For day-to-day work on the learning app, don't start this by hand — run
`delta_drills_local`, which brings up the database, this API and the frontend
together and refuses to hand you a URL until it has proved the auth path works.
See `../LOCAL_DEV.md`.

The steps below are the manual equivalent.

🔴 **`/health` is not a readiness check.** It only reads JSON off disk, so it
answers 200 with the database completely gone — `app/lifecycle.py` catches a
failed schema bootstrap on purpose and only logs a warning. Every endpoint that
touches Postgres 500s while `/health` stays green. To tell whether the database
is really there, POST `/auth/login` with credentials that do not exist and
require a **401**: that means the query ran and found nobody. A 500 is the
database. (`scripts/dd_local_db.sh status` answers the same question from the
other side.)

1) Create and activate a virtualenv.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Set environment variables (copy `.env.example`).

4) Start a database and initialize it. `DATABASE_URL` points at
`localhost:54322` out of the box; `../scripts/dd_local_db.sh up` puts a
container there and waits for it. The schema itself is created on startup by
`app/lifecycle.py` (`Base.metadata.create_all`), so this is only needed for the
older job/chapter tables:

```bash
../scripts/dd_local_db.sh up
python scripts/init_db.py
```

5) Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment

- `DATABASE_URL` supports local PostgreSQL and Supabase's connection string.
  Locally it is a scratch Postgres managed by `../scripts/dd_local_db.sh`, which
  reads this value rather than repeating it — change the port here and the
  container follows on the next `up`.
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
