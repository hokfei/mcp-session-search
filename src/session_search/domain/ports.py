"""Ports (interfaces) that infrastructure adapters must satisfy.

The application layer depends only on these Protocols. Concrete adapters
(SQLite reader, vector index, local model) live in the infrastructure layer.
This is the Dependency Inversion boundary - see ADR 003.
"""

from typing import Protocol, runtime_checkable

from session_search.domain.errors import (
    EmbeddingError,
    RepositoryError,
    Result,
)
from session_search.domain.model import (
    ChunkVector,
    HealthReport,
    IndexStats,
    MessageRecord,
    SearchOutcome,
    SessionMatch,
    SessionRecord,
)
from session_search.domain.value_objects import EmbeddingVector


@runtime_checkable
class SessionSource(Protocol):
    """Read-only source of OpenCode conversation history."""

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        directory: str | None = None,
    ) -> Result[tuple[SessionRecord, ...], RepositoryError]:
        """List sessions, most recently updated first."""
        ...

    async def get_session(self, session_id: str) -> Result[SessionRecord, RepositoryError]:
        """Get one session by ID."""
        ...

    async def get_session_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> Result[tuple[MessageRecord, ...], RepositoryError]:
        """Get a session's messages in chronological order."""
        ...

    async def get_session_transcript(
        self,
        session_id: str,
        max_chars: int = 4000,
    ) -> Result[str, RepositoryError]:
        """Get a session's messages rendered as readable text."""
        ...

    async def get_sessions_since(
        self,
        timestamp_ms: int,
        limit: int = 500,
    ) -> Result[tuple[SessionRecord, ...], RepositoryError]:
        """Get sessions updated after a timestamp, for incremental indexing."""
        ...

    async def count_sessions(self) -> Result[int, RepositoryError]:
        """Total number of sessions available."""
        ...

    async def get_directories(self) -> Result[tuple[str, ...], RepositoryError]:
        """Project directories that have sessions."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores session embeddings and performs similarity search."""

    async def upsert_chunks(
        self,
        session_id: str,
        chunks: tuple[ChunkVector, ...],
        title: str | None,
        directory: str | None,
        time_created: int,
        time_updated: int,
        message_count: int,
        preview: str | None,
    ) -> Result[None, RepositoryError]:
        """Insert or replace a session's chunk vectors and metadata."""
        ...

    async def search(
        self,
        query_embedding: bytes,
        top_k: int = 5,
        threshold: float = 0.35,
        directory: str | None = None,
    ) -> Result[tuple[SessionMatch, ...], RepositoryError]:
        """Rank stored sessions by similarity, one result per session."""
        ...

    async def get_indexed_ids(self) -> Result[set[str], RepositoryError]:
        """IDs of all indexed sessions."""
        ...

    async def get_last_indexed_time(self) -> Result[int, RepositoryError]:
        """Newest time_updated across indexed sessions."""
        ...

    async def count(self) -> Result[int, RepositoryError]:
        """Number of indexed sessions."""
        ...

    async def clear(self) -> Result[None, RepositoryError]:
        """Delete all indexed data."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a normalized vector."""

    async def embed(self, text: str) -> Result[EmbeddingVector, EmbeddingError]:
        """Generate a pooled embedding for text."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector width."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying model is resident in memory."""
        ...


@runtime_checkable
class AnswerWriter(Protocol):
    """Writes a human-readable answer from retrieved session excerpts."""

    @property
    def is_available(self) -> bool:
        """Whether a chat model is configured and loadable."""
        ...

    async def synthesize(self, query: str, context: str) -> str | None:
        """Write an answer, or None when synthesis is unavailable."""
        ...

    def compose_listing(self, query: str, matches: tuple[SessionMatch, ...]) -> str:
        """Build a plain listing of matches without an LLM."""
        ...


__all__ = [
    "AnswerWriter",
    "Embedder",
    "HealthReport",
    "IndexStats",
    "MessageRecord",
    "SearchOutcome",
    "SessionSource",
    "VectorStore",
]
