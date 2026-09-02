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

CREATE TABLE IF NOT EXISTS job_artifacts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  artifact_key text NOT NULL,
  artifact_kind text NOT NULL DEFAULT 'text',
  file_path text,
  content_text text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (job_id, artifact_key)
);

CREATE INDEX IF NOT EXISTS job_artifacts_job_id_idx ON job_artifacts(job_id);

-- ── Study groups ───────────────────────────────────────────────────────────
-- The Groups tab: a handful of learners reading each other's area mastery side
-- by side. Ported from Delta Note's accountability groups. Created at startup
-- by `Base.metadata.create_all` (app/lifecycle.py) as well; this is the
-- out-of-band copy, and app/models.py is the definition the app actually uses.
--
-- 🔴 join_token is a CAPABILITY: anyone holding it is in the group and can read
-- every member's mastery. Rotatable by the owner without touching the roster.

CREATE TABLE IF NOT EXISTS study_groups (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(120) NOT NULL,
  join_token varchar(64) UNIQUE NOT NULL,
  owner_user_id uuid NOT NULL REFERENCES users(id),
  visibility varchar(16) NOT NULL DEFAULT 'private',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS study_groups_join_token_idx ON study_groups(join_token);

CREATE TABLE IF NOT EXISTS study_group_members (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  group_id uuid NOT NULL REFERENCES study_groups(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES users(id),
  display_name varchar(120) NOT NULL,
  joined_at timestamptz NOT NULL DEFAULT now(),
  -- ONE GROUP PER PERSON: "your group" is singular on every surface that reads
  -- it, and this is what makes two racing join clicks produce one row.
  CONSTRAINT study_group_members_user_id_key UNIQUE (user_id)
);

CREATE INDEX IF NOT EXISTS study_group_members_group_id_idx ON study_group_members(group_id);

-- The group's day checklists — one three-state Tiptap list per person, per day.
--
-- 🔴 NO group_id, on purpose. A checklist is something a learner wrote, so it
-- is keyed by (user_id, day) and follows them out of one group and into the
-- next; the group only decides who may READ it (app/study_group_days.py).
--
-- 🔴 `day` is the LEARNER'S LOCAL DATE as their browser named it, never derived
-- from a server timestamp. A day written under the wrong key does not fail — it
-- reads back empty, which is indistinguishable from an empty list.

CREATE TABLE IF NOT EXISTS study_group_days (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id),
  day date NOT NULL,
  -- The Tiptap {v, doc} JSON string. text, not jsonb: round-tripped, never
  -- queried, which is what lets the document schema change without a migration.
  payload text NOT NULL DEFAULT '',
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT study_group_days_user_day_key UNIQUE (user_id, day)
);

CREATE INDEX IF NOT EXISTS study_group_days_user_id_idx ON study_group_days(user_id);
