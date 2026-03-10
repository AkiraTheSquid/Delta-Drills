CREATE TABLE IF NOT EXISTS public.job_artifacts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id uuid NOT NULL REFERENCES public.jobs(id) ON DELETE CASCADE,
  artifact_key text NOT NULL,
  artifact_kind text NOT NULL DEFAULT 'text',
  file_path text,
  content_text text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (job_id, artifact_key)
);

CREATE INDEX IF NOT EXISTS job_artifacts_job_id_idx ON public.job_artifacts(job_id);
