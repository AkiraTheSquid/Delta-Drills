-- Auth and job tables for the deployed backend (Fly.io).
-- These mirror the local PostgreSQL schema so the FastAPI backend
-- can use Supabase PostgreSQL as its database in production.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email text UNIQUE NOT NULL,
  password_hash text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  openai_api_key text,
  mathpix_app_id text,
  mathpix_app_key text
);

CREATE TABLE IF NOT EXISTS jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  original_filename text NOT NULL,
  status text NOT NULL DEFAULT 'queued',
  pdf_path text NOT NULL,
  toc_csv_path text,
  chapters_csv_path text,
  chapters_dir text,
  error_message text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS jobs_user_id_idx ON jobs(user_id);

CREATE TABLE IF NOT EXISTS chapters (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  title text NOT NULL,
  start_page integer NOT NULL,
  end_page integer NOT NULL,
  filename text NOT NULL,
  file_path text NOT NULL,
  file_size integer NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chapters_job_id_idx ON chapters(job_id);
