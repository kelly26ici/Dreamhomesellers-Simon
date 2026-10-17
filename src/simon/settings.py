from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class PoolSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    supabase_url: str = ""
    supabase_key: SecretStr = SecretStr("")
    # The checkpointer's Postgres DSN. Render's dashboard historically names
    # this variable DATABASE_URL, and the deployed Simon service DOES define
    # DATABASE_URL and DOES NOT define POSTGRES_DSN — so reading only
    # POSTGRES_DSN silently yields an empty string and the checkpointer
    # "degrades" for a reason that never appears in the code. Both names are
    # accepted here, POSTGRES_DSN winning on purpose (it is the explicit one).
    postgres_dsn: SecretStr = SecretStr("")
    database_url: SecretStr = SecretStr("")
    # Supabase pooler coordinates. Read so the checkpointer can assemble a
    # reachable DSN when the configured one points at the IPv6-only direct host
    # (db.<ref>.supabase.co has no A record on any Supabase plan; an IPv4
    # address for it is a Pro-plan add-on).
    supabase_pooler_host: str = ""
    supabase_pooler_port: str = ""
    supabase_pooler_user: str = ""
    # Supabase ships this DSN, and it is the only place the pooler-form username
    # + password appear together — which is what a pooler connection must send.
    supabase_pooler_url: SecretStr = SecretStr("")
    supabase_db_host: str = ""
    supabase_db_password: SecretStr = SecretStr("")
    supabase_db_name: str = ""
    supabase_project_region: str = ""
    qdrant_url: str = ""
    qdrant_api_key: SecretStr = SecretStr("")
    cloudflare_account_id: str = ""
    cloudflare_api_token: SecretStr = SecretStr("")

    @property
    def resolved_postgres_dsn(self) -> str:
        """POSTGRES_DSN if set, else DATABASE_URL, else empty.

        Resolved explicitly rather than with a pydantic ``validation_alias``:
        the two names mean the same thing here, but which one the platform
        injects is an accident of deploy history — and a silent empty default is
        the least debuggable of the possibilities.
        """
        for name in ("postgres_dsn", "database_url"):
            value = getattr(self, name)
            if value is not None and value.get_secret_value():
                return value.get_secret_value()
        return ""


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
