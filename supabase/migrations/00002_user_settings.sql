-- User settings: stores per-user configuration such as the OpenAI API key.
-- Row-level security ensures each user can only access their own row.

CREATE TABLE user_settings (
    user_email TEXT PRIMARY KEY,
    openai_api_key TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "read own" ON user_settings
    FOR SELECT USING (user_email = auth.email());

CREATE POLICY "insert own" ON user_settings
    FOR INSERT WITH CHECK (user_email = auth.email());

CREATE POLICY "update own" ON user_settings
    FOR UPDATE USING (user_email = auth.email());
