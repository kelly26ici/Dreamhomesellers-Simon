from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class PoolSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    supabase_url: str = ""
    supabase_key: SecretStr = SecretStr("")
    postgres_dsn: SecretStr = SecretStr("")
    # Supabase pooler coordinates. These already exist in the deployed
    # environment; they are read here only so the checkpointer can assemble a
    # reachable DSN when the configured one points at the IPv6-only direct host.
    supabase_pooler_host: str = ""
    supabase_pooler_port: str = ""
    supabase_pooler_user: str = ""
    supabase_db_host: str = ""
    supabase_db_password: SecretStr = SecretStr("")
    supabase_db_name: str = ""
    supabase_project_region: str = ""
    qdrant_url: str = ""
    qdrant_api_key: SecretStr = SecretStr("")
    cloudflare_account_id: str = ""
    cloudflare_api_token: SecretStr = SecretStr("")


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_model: str = "poolside/laguna-s-2.1"
    nvidia_api_key: SecretStr = SecretStr("")
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    gemini_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    poolside_api_key: SecretStr = SecretStr("")
    poolside_base_url: str = "https://api.poolside.ai/v1"
    openrouter_api_key: SecretStr = SecretStr("")


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "simon"


@lru_cache
def get_pool_settings() -> PoolSettings:
    return PoolSettings()


@lru_cache
def get_model_settings() -> ModelSettings:
    return ModelSettings()


@lru_cache
def get_observability_settings() -> ObservabilitySettings:
    return ObservabilitySettings()
