"""Persistence adapters: reading OpenCode's database and owning our vector index."""

from session_search.infrastructure.persistence.opencode import (
    OpenCodeReader,
    epoch_ms_to_iso,
)
from session_search.infrastructure.persistence.vector_index import (
    VectorIndex,
    cosine_similarity,
)

__all__ = [
    "OpenCodeReader",
    "VectorIndex",
    "cosine_similarity",
    "epoch_ms_to_iso",
]
