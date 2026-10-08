from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    agent_orchestrator_url: str = "http://agent-orchestrator:8004"
    orchestrator_path: str = "/analyze"
    orchestrator_stream_path: str = "/analyze/stream"
    request_timeout_s: float = 30.0
    max_upload_mb: int = 10
    max_prompt_chars: int = 1000
    # Permissif pour le développement local (page de test HTML servie en file://
    # ou http://localhost:*). À restreindre à l'URL du vrai frontend en production.
    cors_allow_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
