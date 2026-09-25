from typing import Optional

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Gloo AI Studio
    gloo_client_id: str
    gloo_client_secret: str
    gloo_token_url: str = "https://platform.ai.gloo.com/oauth2/token"
    gloo_api_base: str = "https://platform.ai.gloo.com/ai/v2"
    gloo_model: str = "gloo-openai-gpt-4.1"
    gloo_scope: str = "api/access"

    # MySQL
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str
    mysql_password: str
    mysql_database: str


    # Redis Cache
    redis_enabled:  bool = False
    redis_host:     str = "localhost"
    redis_port:     int = 6379
    redis_db:       int = 0
    redis_password: str = ""   
    redis_key_prefix: str = ""   

    # Dynamic Schema
    use_dynamic_schema: bool = False
    dynamic_allowed_tables: str = ""   

    # App
    app_env: str = "development"
    app_secret_key: str = "changeme"
    reporting_timezone: str = "Asia/Calcutta"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_dynamic_table_list(self) -> list[str]:
        """Parse comma-separated table names from env."""
        if not self.dynamic_allowed_tables:
            return []
        return [t.strip() for t in self.dynamic_allowed_tables.split(",") if t.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
