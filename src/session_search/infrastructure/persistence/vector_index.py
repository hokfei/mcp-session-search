"""Vector index for OpenCode sessions.

Stores embeddings and metadata in our own SQLite database. Cosine similarity is
computed in pure Python, which avoids the sqlite-vec native extension
(unavailable on this platform) and keeps the dependency surface small.

A session is stored as several **chunks**, each with its own embedding, because
bge-small only accepts short passages. Storing one vector per session forced a
truncation, which made everything past the first ~1500 characters invisible to
search - unacceptable for long-running sessions, where the recent work is the
part worth recalling.

Search scores every chunk, then collapses to one result per session using each
session's best-scoring chunk. A session therefore matches on any part of itself.

At ~10k sessions x 384 dims the linear scan is still a few milliseconds. Beyond
that, swap the scoring loop for HNSW without changing the `VectorStore` port.
"""

import math
import sqlite3
import struct
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from session_search.domain.errors import RepositoryError, Result
from session_search.domain.model import ChunkVector, SessionMatch


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
    """Stores per-chunk session embeddings with pure-Python cosine search."""

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
                    chunk_count INTEGER DEFAULT 1,
                    indexed_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_chunks (
                    session_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding BLOB,
                    text TEXT,
                    PRIMARY KEY (session_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_indexed_time
                    ON indexed_sessions(time_updated DESC);

                CREATE INDEX IF NOT EXISTS idx_indexed_dir
                    ON indexed_sessions(directory);
                """
            )
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Bring an index written by an earlier version up to date."""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(indexed_sessions)")}

        if "embedding" in columns:
            # Pre-chunking layout: move the single stored vector into the
            # chunk table so previously indexed sessions stay searchable.
            conn.execute(
                """
                INSERT OR IGNORE INTO session_chunks (session_id, chunk_index, embedding)
                SELECT session_id, 0, embedding
                FROM indexed_sessions
                WHERE embedding IS NOT NULL
                """
            )
            conn.execute("ALTER TABLE indexed_sessions DROP COLUMN embedding")

        if "chunk_count" not in columns:
            conn.execute("ALTER TABLE indexed_sessions ADD COLUMN chunk_count INTEGER DEFAULT 1")

        # A chunk table written before excerpt storage gained the text column.
        chunk_columns = {row["name"] for row in conn.execute("PRAGMA table_info(session_chunks)")}
        if chunk_columns and "text" not in chunk_columns:
            conn.execute("ALTER TABLE session_chunks ADD COLUMN text TEXT")

        conn.execute(
            """
            UPDATE indexed_sessions
            SET chunk_count = (
                SELECT COUNT(*) FROM session_chunks c
                WHERE c.session_id = indexed_sessions.session_id
            )
            WHERE EXISTS (
                SELECT 1 FROM session_chunks c
                WHERE c.session_id = indexed_sessions.session_id
            )
            """
        )

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
        """Insert or replace a session's chunks and metadata."""
        try:
            with self._conn() as conn:
                conn.execute("DELETE FROM session_chunks WHERE session_id = ?", (session_id,))
                conn.executemany(
                    """
                    INSERT INTO session_chunks (session_id, chunk_index, embedding, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    [(session_id, c.index, c.embedding, c.text) for c in chunks],
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO indexed_sessions
                    (session_id, title, directory, time_created, time_updated,
                     message_count, preview, chunk_count, indexed_at)
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
                        len(chunks),
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
        """Rank sessions by their best-matching chunk."""
        try:
            with self._conn() as conn:
                if directory:
                    rows = conn.execute(
                        """
                        SELECT s.session_id, s.title, s.directory, s.time_created,
                               s.message_count, s.preview,
                               c.chunk_index, c.embedding, c.text
                        FROM session_chunks c
                        JOIN indexed_sessions s ON s.session_id = c.session_id
                        WHERE s.directory = ? AND c.embedding IS NOT NULL
                        """,
                        (directory,),
                    )
                else:
                    rows = conn.execute(
                        """
                        SELECT s.session_id, s.title, s.directory, s.time_created,
                               s.message_count, s.preview,
                               c.chunk_index, c.embedding, c.text
                        FROM session_chunks c
                        JOIN indexed_sessions s ON s.session_id = c.session_id
                        WHERE c.embedding IS NOT NULL
                        """
                    )

                # Best score wins: a session matches on any of its chunks, and
                # the winning chunk's text is the part worth showing.
                best: dict[str, tuple[float, sqlite3.Row]] = {}
                for row in rows:
                    score = cosine_similarity(query_embedding, row["embedding"])
                    if score < threshold:
                        continue
                    current = best.get(row["session_id"])
                    if current is None or score > current[0]:
                        best[row["session_id"]] = (score, row)

            ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)

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
                        matched_chunk_index=row["chunk_index"],
                        matched_text=row["text"],
                    )
                    for score, row in ranked[:top_k]
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

    async def count_chunks(self) -> Result[int, RepositoryError]:
        """Total number of stored chunk vectors."""
        try:
            with self._conn() as conn:
                return Result.ok(conn.execute("SELECT COUNT(*) FROM session_chunks").fetchone()[0])
        except Exception as e:
            return Result.err(RepositoryError(f"Failed to count chunks: {e}", e))

    async def remove(self, session_id: str) -> Result[None, RepositoryError]:
        """Delete one session and all of its chunk vectors."""
        try:
            with self._conn() as conn:
                conn.execute("DELETE FROM session_chunks WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM indexed_sessions WHERE session_id = ?", (session_id,))
            return Result.ok(None)
        except Exception as e:
            return Result.err(RepositoryError(f"Failed to remove session: {e}", e))

    async def prune_missing(self, live_ids: set[str]) -> Result[int, RepositoryError]:
        """Drop index entries whose sessions no longer exist.

        Without this, a session deleted from OpenCode keeps its embeddings and
        full transcript text in our database indefinitely, and nothing in the
        system could remove them. Returns how many sessions were dropped.
        """
        try:
            with self._conn() as conn:
                indexed = {
                    row[0] for row in conn.execute("SELECT session_id FROM indexed_sessions")
                }
                orphans = indexed - live_ids

                for session_id in orphans:
                    conn.execute("DELETE FROM session_chunks WHERE session_id = ?", (session_id,))
                    conn.execute("DELETE FROM indexed_sessions WHERE session_id = ?", (session_id,))

            return Result.ok(len(orphans))
        except Exception as e:
            return Result.err(RepositoryError(f"Failed to prune stale sessions: {e}", e))

    async def clear(self) -> Result[None, RepositoryError]:
        """Delete all indexed data."""
        try:
            with self._conn() as conn:
                conn.execute("DELETE FROM session_chunks")
                conn.execute("DELETE FROM indexed_sessions")
            return Result.ok(None)
        except Exception as e:
            return Result.err(RepositoryError(f"Failed to clear index: {e}", e))


__all__ = ["ChunkVector", "VectorIndex", "cosine_similarity"]
