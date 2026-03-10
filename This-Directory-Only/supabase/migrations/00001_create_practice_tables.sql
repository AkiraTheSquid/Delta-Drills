-- Practice user state: stores the full adaptive state as a single JSONB blob
-- per user. Matches the JSON file structure used by the local backend.

CREATE TABLE practice_user_state (
    user_email TEXT PRIMARY KEY,
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE practice_user_state ENABLE ROW LEVEL SECURITY;

-- RLS policies: users can only access their own row
CREATE POLICY "read own" ON practice_user_state
    FOR SELECT USING (user_email = auth.email());

CREATE POLICY "insert own" ON practice_user_state
    FOR INSERT WITH CHECK (user_email = auth.email());

CREATE POLICY "update own" ON practice_user_state
    FOR UPDATE USING (user_email = auth.email());
