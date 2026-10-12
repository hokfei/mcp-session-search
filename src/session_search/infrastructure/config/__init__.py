"""Infrastructure configuration."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def home_dir() -> Path:
    """Resolve the user's home directory without ever raising.

    `Path.home()` raises when neither HOME nor USERPROFILE is set, which some
    MCP launchers do. Falling back to the current directory keeps the server
    importable; the paths simply need explicit configuration in that case.
    """
    try:
        return Path.home()
    except (RuntimeError, KeyError):
        for variable in ("HOME", "USERPROFILE"):
            value = os.environ.get(variable)
            if value:
                return Path(value)
        return Path.cwd()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Without this, pydantic reads bare names like MODEL_PATH and any
        # session_search_* variable set by the MCP client is silently ignored.
        env_prefix="session_search_",
    )

    # ==================== Storage ====================

    database_path: Path = Field(
        default_factory=lambda: home_dir() / ".opencode" / "session_search.db",
        description="SQLite database for the vector index",
    )

    opencode_db_path: Path = Field(
        default_factory=lambda: home_dir() / ".local" / "share" / "opencode" / "opencode.db",
        description="OpenCode's own session database (read-only)",
    )

    # ==================== Embedding Model ====================

    model_path: Path = Field(
        default_factory=lambda: home_dir() / "models" / "bge-small-en-v1.5-q8_0.gguf",
        description=(
            "Embedding model GGUF. bge-small-en-v1.5 produces "
            "fixed 384-dim pooled vectors (~36MB, fast on CPU)."
        ),
    )

    model_n_ctx: int = Field(
        default=512,
        description="Context window (bge-small trained at 512 tokens)",
    )

    model_n_gpu_layers: int = Field(
        default=0,
        description="GPU layers (-1 = all, 0 = CPU only)",
    )

    model_n_threads: int = Field(
        default=0,
        description="CPU threads (0 = auto-detect)",
    )

    embedding_dim: int = Field(
        default=384,
        description="Embedding dimension (bge-small-en-v1.5 = 384)",
    )

    # ==================== Chat Model (answer synthesis) ====================

    chat_model_path: Path | None = Field(
        default=None,
        description=(
            "Optional chat model GGUF for writing natural-language answers. "
            "When None, search returns raw session matches without synthesis."
        ),
    )

    chat_model_n_ctx: int = Field(
        default=2048,
        description="Context window for the chat model (answer synthesis)",
    )

    max_summary_tokens: int = Field(
        default=150,
        description=(
            "Max tokens for synthesized answers. Measured on Qwen3.5-4B at "
            "roughly 5 tok/s on CPU, so this is about a 30s ceiling; real "
            "answers are 20-40 tokens."
        ),
    )

    # ==================== Search ====================

    default_top_k: int = Field(
        default=5,
        description="Default number of search results",
    )

    max_top_k: int = Field(
        default=20,
        description="Maximum search results allowed",
    )

    similarity_threshold: float = Field(
        default=0.55,
        description=(
            "Minimum cosine similarity for results. Calibrated against the "
            "local session corpus: real matches score 0.60-0.85 at rank 1, "
            "unrelated queries top out around 0.535."
        ),
    )

    # ==================== Indexing ====================

    auto_index_on_search: bool = Field(
        default=True,
        description="Index new/changed sessions automatically when searching",
    )

    index_max_transcript_chars: int = Field(
        default=1500,
        description="Max transcript chars embedded per session",
    )

    index_batch_size: int = Field(
        default=1,
        description="Sessions to embed before yielding (1 = safest for CPU)",
    )

    # ==================== Server ====================

    server_name: str = Field(
        default="session-search",
        description="MCP server name",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )


# Global settings instance
settings = Settings()
