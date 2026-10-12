"""Domain layer: the model, the ports, and the errors.

This layer has no knowledge of SQLite, llama.cpp, or MCP. Everything the
application layer needs is declared here and implemented in infrastructure.
"""

from session_search.domain.errors import (
    ConfigurationError,
    DomainError,
    EmbeddingError,
    ModelLoadError,
    RepositoryError,
    Result,
    SearchError,
    SummarizationError,
    SystemError,
)
from session_search.domain.model import (
    HealthReport,
    IndexStats,
    MessageRecord,
    SearchOutcome,
    SessionMatch,
    SessionRecord,
)
from session_search.domain.ports import (
    AnswerWriter,
    Embedder,
    SessionSource,
    VectorStore,
)
from session_search.domain.value_objects import EmbeddingVector, SimilarityScore

__all__ = [
    # Errors
    "ConfigurationError",
    "DomainError",
    "EmbeddingError",
    "ModelLoadError",
    "RepositoryError",
    "Result",
    "SearchError",
    "SummarizationError",
    "SystemError",
    # Model
    "HealthReport",
    "IndexStats",
    "MessageRecord",
    "SearchOutcome",
    "SessionMatch",
    "SessionRecord",
    # Ports
    "AnswerWriter",
    "Embedder",
    "SessionSource",
    "VectorStore",
    # Value objects
    "EmbeddingVector",
    "SimilarityScore",
]
