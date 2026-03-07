-- Add Mathpix credentials to user_settings table.

ALTER TABLE user_settings
    ADD COLUMN IF NOT EXISTS mathpix_app_id TEXT,
    ADD COLUMN IF NOT EXISTS mathpix_app_key TEXT;
