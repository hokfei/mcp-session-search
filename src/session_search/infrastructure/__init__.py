"""Infrastructure layer: adapters implementing the domain ports.

Concrete adapters for OpenCode's SQLite database, our vector index, and local
GGUF models. This layer is the only place allowed to know about SQL, file
paths, and llama.cpp.
"""

from session_search.infrastructure.config import Settings, settings
from session_search.infrastructure.llm import LocalAnswerWriter, LocalEmbedder
from session_search.infrastructure.persistence.opencode import (
    OpenCodeReader,
    epoch_ms_to_iso,
)
from session_search.infrastructure.persistence.vector_index import (
    VectorIndex,
    cosine_similarity,
)

__all__ = [
    "LocalAnswerWriter",
    "LocalEmbedder",
    "OpenCodeReader",
    "Settings",
    "VectorIndex",
    "cosine_similarity",
    "epoch_ms_to_iso",
    "settings",
]
