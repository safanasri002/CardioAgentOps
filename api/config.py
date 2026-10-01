from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    audio_engine_url: str = "http://audio-engine:8003"
    agent_orchestrator_url: str = "http://agent-orchestrator:8004"
    audio_engine_path: str = "/predict"
    orchestrator_path: str = "/analyze"
    request_timeout_s: float = 30.0
    max_upload_mb: int = 10
    # Permissif pour le développement local (page de test HTML servie en file://
    # ou http://localhost:*). À restreindre à l'URL du vrai frontend en production.
    cors_allow_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
