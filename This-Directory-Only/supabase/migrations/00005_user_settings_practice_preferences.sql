-- Persist per-user practice/statistics preferences in Supabase so the
-- deployed app doesn't depend on localStorage-only state.

ALTER TABLE user_settings
    ADD COLUMN IF NOT EXISTS practice_preferences JSONB NOT NULL DEFAULT '{}'::jsonb;
