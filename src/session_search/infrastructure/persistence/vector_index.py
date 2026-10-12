"""Vector index for OpenCode sessions.

Keeps embeddings and search metadata in our own SQLite database. Cosine
similarity is computed in pure Python, which avoids the sqlite-vec native
extension (unavailable on this platform) and keeps the dependency surface small.

At ~10k sessions x 384 dims a linear scan is a few milliseconds. If the index
ever grows an order of magnitude larger, swap the scoring loop for HNSW without
changing the `VectorStore` port.
"""

import math
import sqlite3
import struct
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from session_search.domain.errors import RepositoryError, Result
from session_search.domain.model import SessionMatch


def cosine_similarity(a: bytes, b: bytes) -> float:
    """Cosine similarity between two serialized float32 vectors."""
    if len(a) != len(b):
        return 0.0

    count = len(a) // 4
    left = struct.unpack(f"{count}f", a)
    right = struct.unpack(f"{count}f", b)

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(left, right, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class VectorIndex:
    """Stores session embeddings with pure-Python cosine search."""

    def __init__(self, db_path: Path, embedding_dim: int = 384) -> None:
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS indexed_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    directory TEXT,
                    time_created INTEGER NOT NULL,
                    time_updated INTEGER NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    preview TEXT,
                    embedding BLOB,
                    indexed_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_indexed_time
                    ON indexed_sessions(time_updated DESC);

                CREATE INDEX IF NOT EXISTS idx_indexed_dir
                    ON indexed_sessions(directory);
                """
            )

    async def upsert(
        self,
        session_id: str,
        embedding: bytes,
        title: str | None,
        directory: str | None,
        time_created: int,
        time_updated: int,
        message_count: int,
        preview: str | None,
    ) -> Result[None, RepositoryError]:
        """Insert or replace a session's vector and metadata."""
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO indexed_sessions
                    (session_id, title, directory, time_created, time_updated,
                     message_count, preview, embedding, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        title,
                        directory,
                        time_created,
                        time_updated,
                        message_count,
                        preview,
                        embedding,
                        int(time.time() * 1000),
                    ),
                )
            return Result.ok(None)

        except Exception as e:
            return Result.err(RepositoryError(f"Failed to index session: {e}", e))

    async def search(
        self,
        query_embedding: bytes,
        top_k: int = 5,
        threshold: float = 0.35,
        directory: str | None = None,
    ) -> Result[tuple[SessionMatch, ...], RepositoryError]:
        """Rank indexed sessions by cosine similarity to the query vector."""
        try:
            with self._conn() as conn:
                if directory:
                    cursor = conn.execute(
                        """
                        SELECT session_id, title, directory, time_created,
                               message_count, preview, embedding
                        FROM indexed_sessions
                        WHERE directory = ? AND embedding IS NOT NULL
                        """,
                        (directory,),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT session_id, title, directory, time_created,
                               message_count, preview, embedding
                        FROM indexed_sessions
                        WHERE embedding IS NOT NULL
                        """
                    )

                scored: list[tuple[float, sqlite3.Row]] = []
                for row in cursor:
                    score = cosine_similarity(query_embedding, row["embedding"])
                    if score >= threshold:
                        scored.append((score, row))

            scored.sort(key=lambda pair: pair[0], reverse=True)

            return Result.ok(
                tuple(
                    SessionMatch(
                        session_id=row["session_id"],
                        title=row["title"],
                        directory=row["directory"],
                        time_created=row["time_created"],
                        message_count=row["message_count"],
                        preview=row["preview"],
                        score=score,
                    )
                    for score, row in scored[:top_k]
                )
            )

        except Exception as e:
            return Result.err(RepositoryError(f"Search failed: {e}", e))

    async def get_indexed_ids(self) -> Result[set[str], RepositoryError]:
        """IDs of every indexed session."""
        try:
            with self._conn() as conn:
                rows = conn.execute("SELECT session_id FROM indexed_sessions").fetchall()
                return Result.ok({r["session_id"] for r in rows})
        except Exception as e:
            return Result.err(RepositoryError(f"Failed to get indexed IDs: {e}", e))

    async def get_last_indexed_time(self) -> Result[int, RepositoryError]:
        """Newest time_updated across indexed sessions (0 when empty)."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT MAX(time_updated) AS max_time FROM indexed_sessions"
                ).fetchone()
                return Result.ok(row["max_time"] or 0)
        except Exception as e:
            return Result.err(RepositoryError(f"Failed to get last indexed time: {e}", e))

    async def count(self) -> Result[int, RepositoryError]:
        """Number of indexed sessions."""
        try:
            with self._conn() as conn:
                return Result.ok(
                    conn.execute("SELECT COUNT(*) FROM indexed_sessions").fetchone()[0]
                )
        except Exception as e:
            return Result.err(RepositoryError(f"Failed to count indexed sessions: {e}", e))

    async def clear(self) -> Result[None, RepositoryError]:
        """Delete all indexed data."""
        try:
            with self._conn() as conn:
                conn.execute("DELETE FROM indexed_sessions")
            return Result.ok(None)
        except Exception as e:
            return Result.err(RepositoryError(f"Failed to clear index: {e}", e))


__all__ = ["VectorIndex", "cosine_similarity"]
