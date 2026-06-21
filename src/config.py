"""Runtime configuration for Scholar RAG Agent."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="SCHOLAR_RAG_", env_file=".env", extra="ignore")

    database_path: Path = Field(default=Path(".scholar-rag-agent.sqlite3"))
    agent_id: str = Field(default="local-agent")
    retrieval_timeout_seconds: float = Field(default=30.0, ge=1.0)
    reasoning_timeout_seconds: float = Field(default=60.0, ge=1.0)
    max_source_docs: int = Field(default=50, ge=1, le=50)
    max_hops: int = Field(default=5, ge=1, le=5)
    default_model: str = Field(default="openai")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    moonshot_api_key: str | None = Field(default=None, validation_alias="MOONSHOT_API_KEY")
    semantic_scholar_api_key: str | None = Field(
        default=None, validation_alias="SEMANTIC_SCHOLAR_API_KEY"
    )


def load_settings() -> Settings:
    """Return settings from the current process environment."""
    return Settings()
