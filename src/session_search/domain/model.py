"""Domain model for OpenCode session history.

These types describe *what a session is*, independent of where the data comes
from (OpenCode's SQLite database) and how it is stored (our vector index).
Both infrastructure adapters map onto these, so the application layer never
needs to know which is which.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """A conversation session as recorded by OpenCode."""

    id: str
    title: str | None
    directory: str | None
    project_id: str
    version: str
    agent: str | None
    model: str | None
    time_created: int  # epoch milliseconds
    time_updated: int
    message_count: int
    tokens_input: int
    tokens_output: int
    cost: float

    @property
    def created_at(self) -> datetime:
        return datetime.fromtimestamp(self.time_created / 1000)

    @property
    def updated_at(self) -> datetime:
        return datetime.fromtimestamp(self.time_updated / 1000)

    @property
    def has_content(self) -> bool:
        """Whether this session carries enough content to be worth indexing."""
        return self.message_count > 0


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """A single conversational turn within a session."""

    id: str
    session_id: str
    seq: int
    role: str  # user | assistant | system
    text: str
    time_created: int
    metadata: dict[str, Any]

    @property
    def created_at(self) -> datetime:
        return datetime.fromtimestamp(self.time_created / 1000)


@dataclass(frozen=True, slots=True)
class SessionMatch:
    """A session returned by semantic search, with its relevance score."""

    session_id: str
    title: str | None
    directory: str | None
    time_created: int
    message_count: int
    preview: str | None
    score: float

    @property
    def created_at(self) -> datetime:
        return datetime.fromtimestamp(self.time_created / 1000)

    @property
    def display_title(self) -> str:
        return self.title or "untitled session"


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Outcome of a single indexing pass."""

    indexed: int
    skipped: int
    found: int

    @property
    def summary(self) -> str:
        return f"Indexed {self.indexed} of {self.found} sessions ({self.skipped} skipped)."


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Complete result of a search: matches plus a written answer."""

    query: str
    matches: tuple[SessionMatch, ...]
    answer: str
    synthesized: bool

    @property
    def is_empty(self) -> bool:
        return not self.matches

    def __len__(self) -> int:
        return len(self.matches)


@dataclass(frozen=True, slots=True)
class HealthReport:
    """System health snapshot."""

    opencode_db_readable: bool
    opencode_db_path: str
    total_sessions: int
    indexed_sessions: int
    embedding_model: str
    embedding_model_loaded: bool
    answer_synthesis_available: bool

    @property
    def status(self) -> str:
        return "healthy" if self.opencode_db_readable else "unhealthy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "opencode_db": self.opencode_db_path,
            "opencode_db_readable": self.opencode_db_readable,
            "total_sessions_in_opencode": self.total_sessions,
            "indexed_sessions": self.indexed_sessions,
            "embedding_model": self.embedding_model,
            "embedding_model_loaded": self.embedding_model_loaded,
            "answer_synthesis_available": self.answer_synthesis_available,
        }


__all__ = [
    "HealthReport",
    "IndexStats",
    "MessageRecord",
    "SearchOutcome",
    "SessionMatch",
    "SessionRecord",
]
