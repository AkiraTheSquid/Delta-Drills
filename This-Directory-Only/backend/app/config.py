from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:54322/pdf_split_tool"
    jwt_secret: str = "change-me"
    access_token_ttl_minutes: int = 720
    storage_dir: str = "/tmp/delta-drills-local"
    openai_api_key: str | None = None
    openai_model: str | None = None
    mathpix_app_id: str | None = None
    mathpix_app_key: str | None = None
    supabase_url: str = "https://qaxtcaoydbpigomnfjpl.supabase.co"
    supabase_service_role_key: str = ""
    user_data_dir: str = ""  # Override for deployed envs (e.g. /data/user_data on Fly.io)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
