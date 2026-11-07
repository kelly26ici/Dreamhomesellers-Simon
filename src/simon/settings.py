"""Process-wide settings, resolved once.

One Settings class replaces the previous three (PoolSettings, ModelSettings,
ObservabilitySettings). They read the same `.env` file, the same way, so the
split created three independent env parses and three cache keys for what is
conceptually one configuration object -- and it made a setting's home
ambiguous, which is how `luxonn_site_url` ended up in the pool group despite
having nothing to do with pooling.

Every environment variable name from the previous classes is preserved
verbatim, so nothing in the Render dashboard or `.env` needs to change.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# (env key for the API key, default model id) in FALLBACK ORDER. A provider
# with no key configured is skipped, never a runtime failure.
MODEL_PROVIDER_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
    ("poolside", "laguna-s-2.1"),
    ("groq", "openai/gpt-oss-120b"),
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Supabase / database -------------------------------------------
    supabase_url: str = ""
    supabase_key: SecretStr = SecretStr("")
    # The checkpointer's Postgres DSN. Render's dashboard historically names
    # this DATABASE_URL, and the deployed Simon service DOES define
    # DATABASE_URL and DOES NOT define POSTGRES_DSN -- so reading only
    # POSTGRES_DSN silently yields an empty string and the checkpointer
    # "degrades" for a reason that never appears in the code. Both names are
    # accepted, POSTGRES_DSN winning on purpose (it is the explicit one).
    postgres_dsn: SecretStr = SecretStr("")
    database_url: SecretStr = SecretStr("")
    # Supabase pooler coordinates. Read so the checkpointer can assemble a
    # reachable DSN when the configured one points at the IPv6-only direct
    # host (db.<ref>.supabase.co has no A record on any Supabase plan; an IPv4
    # address for it is a Pro-plan add-on). Note for anyone reading this on a
    # machine with no IPv6: none of these hostnames resolve there, which is
    # why production migrations are applied from the Supabase dashboard.
    supabase_pooler_host: str = ""
    supabase_pooler_port: str = ""
    supabase_pooler_user: str = ""
    supabase_pooler_url: SecretStr = SecretStr("")
    supabase_db_host: str = ""
    supabase_db_password: SecretStr = SecretStr("")
    supabase_db_name: str = ""
    supabase_project_region: str = ""

    # ---- Vector search --------------------------------------------------
    qdrant_url: str = ""
    qdrant_api_key: SecretStr = SecretStr("")
    cloudflare_account_id: str = ""
    cloudflare_api_token: SecretStr = SecretStr("")

    # ---- Model providers (fallback order is MODEL_PROVIDER_DEFAULTS) -----
    nvidia_api_key: SecretStr = SecretStr("")
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    gemini_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    poolside_api_key: SecretStr = SecretStr("")
    poolside_base_url: str = "https://api.poolside.ai/v1"
    openrouter_api_key: SecretStr = SecretStr("")
    # The historical primary-model override. Kept so an existing deployment
    # that sets it keeps working; ordering now lives in
    # MODEL_PROVIDER_DEFAULTS and is serialised by `model_chain()`.
    llm_model: str = "poolside/laguna-s-2.1"

    # ---- Service surface -------------------------------------------------
    # The companion website's Render base URL. Read by keepalive.py to give
    # that service inbound traffic -- the only thing that keeps a free Render
    # web service from spinning down. Empty = keep-alive disabled.
    luxonn_site_url: str = ""
    # CORS. Empty means same-origin only, which is the correct default in
    # production: this service is called by the website's server-side relay,
    # not by browsers directly. The previous hardcoded localhost list was both
    # dead in production and a deploy blocker.
    cors_allowed_origins: str = ""
    # Shared secret callers must present as a header. Without it every
    # endpoint -- including the bounded /api/aux model call -- is an open LLM
    # proxy on a public host. Empty = authentication disabled, which is only
    # acceptable for a local run.
    #
    # Both names are accepted on purpose. The field is `shared_secret`, so
    # pydantic reads `SHARED_SECRET` by default; the website namespaces the same
    # credential as `SIMON_SHARED_SECRET`, and Render carries that name on BOTH
    # services so the two sides agree by construction. Before this alias existed
    # the names could drift, and because the model uses extra="ignore" the
    # mismatch was silent: Simon read an empty string, logged "auth disabled",
    # and enforced nothing while appearing configured.
    shared_secret: SecretStr = SecretStr("")
    simon_shared_secret: SecretStr = SecretStr("")
    port: int = 8787

    # ---- Middleware behaviour (LangChain ModelRetry/ToolRetry) --------------
    # Bound tightly. Every retry is a paid provider call, and a buyer waiting
    # on a turn that is quietly retrying three times is a buyer leaving.
    model_max_retries: int = 2
    model_backoff_factor: float = 2.0
    model_initial_delay: float = 1.0
    model_max_delay: float = 8.0

    # ---- Observability ---------------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "simon"

    # ---- Derived helpers --------------------------------------------------

    @property
    def resolved_postgres_dsn(self) -> str:
        """POSTGRES_DSN if set, else DATABASE_URL, else empty.

        Resolved explicitly rather than with a pydantic ``validation_alias``:
        the two names mean the same thing here, but which one the platform
        injects is an accident of deploy history -- and a silent empty default
        is the least debuggable of the possibilities.
        """
        for name in ("postgres_dsn", "database_url"):
            value = getattr(self, name)
            if value is not None and value.get_secret_value():
                return value.get_secret_value()
        return ""

    @property
    def cors_origins(self) -> list[str]:
        """Parsed CORS_ALLOWED_ORIGINS. Empty list = same-origin only."""
        raw = self.cors_allowed_origins.strip()
        if not raw:
            return []
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    def api_key_for(self, provider: str) -> str:
        """The configured API key for a provider, or empty when unconfigured."""
        key = getattr(self, f"{provider}_api_key", None)
        return key.get_secret_value() if key else ""

    @property
    def shared_secret_value(self) -> str:
        """The shared secret, from either accepted name; empty when unset.

        Both names are read because Render carries `SIMON_SHARED_SECRET` on both
        services while pydantic's default for this field is `SHARED_SECRET`.
        Resolving here, rather than through a `validation_alias`, keeps the
        precedence visible in code instead of implicit in the settings model.
        """
        for name in ("shared_secret", "simon_shared_secret"):
            value = getattr(self, name)
            secret = value.get_secret_value() if value else ""
            if secret:
                return secret
        return ""

    def model_override_for(self, provider: str) -> str:
        """A per-provider model override, e.g. SIMON_GROQ_MODEL, or empty.

        `.env.example` documented these names but no code ever read them;
        they become real here so swapping a dead provider is an env change and
        not a code change.
        """
        return getattr(self, f"simon_{provider}_model", "") or ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The one settings object. Cached once per process, never per request."""
    return Settings()


# Historical name, kept importable. `PoolSettings` was the first of the three
# classes consolidated into `Settings`; the checkpointer tests build instances
# directly with `_env_file=""` to isolate them from the real `.env`, and they
# construct `PoolSettings(...)` by name. Aliasing keeps those tests honest
# instead of rewriting 28 call sites to chase a rename.
PoolSettings = Settings

