"""OpenCode session reader.

Reads conversation history straight out of OpenCode's own SQLite database in
strictly read-only mode. This is what makes the memory useful with zero manual
logging: every session OpenCode has ever recorded is already there.

OpenCode's schema has changed over time. Newer installs keep sessions in
`session_v2` / `session_message`; older ones use `session` / `message`. Both are
supported, and the reader spans them so historical sessions stay searchable.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from session_search.domain.errors import RepositoryError, Result
from session_search.domain.model import MessageRecord, SessionRecord

# Roles that represent an actual conversational turn. Bookkeeping rows such as
# `idle` and `compaction` are excluded from transcripts and message counts.
CONVERSATIONAL_TYPES = ("user", "assistant", "system")

# Columns selected for any session row, shared by every query.
_SESSION_COLUMNS = """
    id, title, directory, project_id, version,
    agent, model, time_created, time_updated,
    tokens_input, tokens_output, cost, time_archived
"""

# OpenCode soft-archives rather than deleting: removing a session from the list
# stamps time_archived and leaves the row and its messages in place. Archived
# sessions are excluded unless explicitly requested, because a session the user
# removed should stop being recallable.
_ARCHIVED_FILTER = "AND time_archived IS NULL"
_ARCHIVED_FILTER_SELECT = "CASE WHEN time_archived IS NULL THEN 0 ELSE 1 END AS archived"


class OpenCodeReader:
    """Read-only access to OpenCode's session database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise RepositoryError(f"OpenCode database not found: {self.db_path}")

    @property
    def is_readable(self) -> bool:
        """Whether the database file is present and openable."""
        return self.db_path.exists()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Open a strictly read-only connection (SQLite URI mode=ro)."""
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ==================== Queries ====================

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        directory: str | None = None,
        include_archived: bool = False,
    ) -> Result[tuple[SessionRecord, ...], RepositoryError]:
        """List sessions across both schema generations, newest first.

        Archived sessions are omitted unless `include_archived` is set.
        """
        try:
            with self._conn() as conn:
                # Over-fetch per table, then dedupe, sort and page in Python so
                # that sessions present in both tables are not double-counted.
                fetch = limit + offset
                rows: list[sqlite3.Row] = []

                for table, count_expr in (
                    ("session_v2", _COUNT_V2),
                    ("session", _COUNT_LEGACY),
                ):
                    if not self._has_table(conn, table):
                        continue

                    archive_clause = "" if include_archived else _ARCHIVED_FILTER
                    rows.extend(
                        conn.execute(
                            f"""
                            SELECT {_SESSION_COLUMNS},
                                   {_ARCHIVED_FILTER_SELECT},
                                   {count_expr} AS message_count
                            FROM {table}
                            WHERE (? IS NULL OR directory = ?) {archive_clause}
                            ORDER BY time_updated DESC
                            LIMIT ?
                            """,
                            (directory, directory, fetch),
                        ).fetchall()
                    )

                records = _dedupe(self._to_session(r) for r in rows)

            ordered = sorted(records, key=lambda s: s.time_updated, reverse=True)
            return Result.ok(tuple(ordered[offset : offset + limit]))

        except Exception as e:
            return Result.err(RepositoryError(f"Failed to list sessions: {e}", e))

    async def get_session(self, session_id: str) -> Result[SessionRecord, RepositoryError]:
        """Get one session by ID from either schema generation.

        Archived sessions are still returned: if the caller already has the ID,
        hiding it would be surprising. Search excludes them.
        """
        try:
            with self._conn() as conn:
                for table, count_expr in (
                    ("session_v2", _COUNT_V2),
                    ("session", _COUNT_LEGACY),
                ):
                    if not self._has_table(conn, table):
                        continue

                    row = conn.execute(
                        f"""
                        SELECT {_SESSION_COLUMNS},
                               {_ARCHIVED_FILTER_SELECT},
                               {count_expr} AS message_count
                        FROM {table}
                        WHERE id = ?
                        """,
                        (session_id,),
                    ).fetchone()

                    if row:
                        return Result.ok(self._to_session(row))

            return Result.err(RepositoryError(f"Session not found: {session_id}"))

        except Exception as e:
            return Result.err(RepositoryError(f"Failed to get session: {e}", e))

    async def get_session_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> Result[tuple[MessageRecord, ...], RepositoryError]:
        """Get a session's conversational messages in chronological order."""
        try:
            with self._conn() as conn:
                query: str
                if self._has_table(conn, "session_message"):
                    query = """
                        SELECT id, session_id, seq, type, data, time_created
                        FROM session_message
                        WHERE session_id = ?
                          AND type IN ('user', 'assistant', 'system')
                        ORDER BY seq ASC
                    """
                else:
                    # Legacy rows carry the role inside the JSON payload and
                    # have no monotonic seq, so rowid stands in for ordering.
                    query = """
                        SELECT id, session_id, rowid AS seq, NULL AS type,
                               data, time_created
                        FROM message
                        WHERE session_id = ?
                        ORDER BY time_created ASC
                    """

                params: list[Any] = [session_id]
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                rows = conn.execute(query, params).fetchall()

                messages = [m for m in (self._to_message(r) for r in rows) if m]
                return Result.ok(tuple(messages))

        except Exception as e:
            return Result.err(RepositoryError(f"Failed to get messages: {e}", e))

    async def get_session_transcript(
        self,
        session_id: str,
        max_chars: int = 4000,
    ) -> Result[str, RepositoryError]:
        """Render a session's messages as plain text for embedding or answering."""
        messages_result = await self.get_session_messages(session_id)
        if not messages_result.is_ok:
            return Result.err(messages_result.error)

        parts: list[str] = []
        for message in messages_result.value:
            text = message.text.strip()
            if not text:
                continue
            # Very long tool outputs would swamp the embedding; cap each message.
            if len(text) > 800:
                text = text[:800].rstrip() + " ..."
            parts.append(f"{message.role}: {text}")

        transcript = "\n".join(parts)
        return Result.ok(transcript[:max_chars])

    async def get_sessions_since(
        self,
        timestamp_ms: int,
        limit: int = 500,
        include_archived: bool = False,
    ) -> Result[tuple[SessionRecord, ...], RepositoryError]:
        """Get sessions updated at or after a timestamp, oldest first.

        Oldest-first lets an interrupted indexing pass resume where it left off.

        The comparison is inclusive (`>=`) on purpose. Callers pass the newest
        `time_updated` they have already indexed, and an exclusive comparison
        would permanently skip any session whose update time exactly matches
        that watermark - including the very session that set it. Re-seeing one
        already-indexed session per pass is harmless because callers skip by ID.
        """
        try:
            with self._conn() as conn:
                rows: list[sqlite3.Row] = []
                archive_clause = "" if include_archived else _ARCHIVED_FILTER

                for table, count_expr in (
                    ("session_v2", _COUNT_V2),
                    ("session", _COUNT_LEGACY),
                ):
                    if not self._has_table(conn, table):
                        continue
                    rows.extend(
                        conn.execute(
                            f"""
                            SELECT {_SESSION_COLUMNS},
                                   {_ARCHIVED_FILTER_SELECT},
                                   {count_expr} AS message_count
                            FROM {table}
                            WHERE time_updated >= ? {archive_clause}
                            ORDER BY time_updated ASC
                            LIMIT ?
                            """,
                            (timestamp_ms, limit),
                        ).fetchall()
                    )

                records = _dedupe(self._to_session(r) for r in rows)

            ordered = sorted(records, key=lambda s: s.time_updated)
            return Result.ok(tuple(ordered[:limit]))

        except Exception as e:
            return Result.err(RepositoryError(f"Failed to get recent sessions: {e}", e))

    async def count_sessions(self) -> Result[int, RepositoryError]:
        """Count sessions across both schema generations, excluding archived.

        IDs may appear in both tables during a migration, so this reports the
        union rather than a naive sum.
        """
        ids = await self.get_all_session_ids()
        if not ids.is_ok:
            return Result.err(ids.error)
        return Result.ok(len(ids.value))

    async def get_all_session_ids(
        self, include_archived: bool = False
    ) -> Result[set[str], RepositoryError]:
        """Every session ID currently stored, for reconciling the index.

        Used to drop index entries whose session no longer exists, which would
        otherwise linger forever with nothing able to remove them.
        """
        try:
            with self._conn() as conn:
                ids: set[str] = set()
                archive_clause = "" if include_archived else _ARCHIVED_FILTER

                for table in ("session_v2", "session"):
                    if not self._has_table(conn, table):
                        continue
                    ids.update(
                        row[0]
                        for row in conn.execute(
                            f"SELECT id FROM {table} WHERE 1=1 {archive_clause}"
                        )
                    )

                return Result.ok(ids)

        except Exception as e:
            return Result.err(RepositoryError(f"Failed to list session IDs: {e}", e))

    async def get_directories(self) -> Result[tuple[str, ...], RepositoryError]:
        """Project directories that contain sessions."""
        try:
            with self._conn() as conn:
                dirs: set[str] = set()

                for table in ("session_v2", "session"):
                    if self._has_table(conn, table):
                        dirs.update(
                            row[0]
                            for row in conn.execute(
                                f"""
                                SELECT DISTINCT directory FROM {table}
                                WHERE directory IS NOT NULL
                                """
                            )
                        )

                return Result.ok(tuple(sorted(dirs)))

        except Exception as e:
            return Result.err(RepositoryError(f"Failed to get directories: {e}", e))

    # ==================== Mapping ====================

    @staticmethod
    def _has_table(conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    @staticmethod
    def _to_session(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            id=row["id"],
            title=row["title"],
            directory=row["directory"],
            project_id=row["project_id"],
            version=row["version"],
            agent=row["agent"],
            model=row["model"],
            time_created=row["time_created"],
            time_updated=row["time_updated"],
            message_count=row["message_count"] or 0,
            tokens_input=row["tokens_input"] or 0,
            tokens_output=row["tokens_output"] or 0,
            cost=row["cost"] or 0.0,
            archived=bool(row["archived"]) if "archived" in row.keys() else False,
        )

    def _to_message(self, row: sqlite3.Row) -> MessageRecord | None:
        """Map one message row, returning None for unusable entries."""
        try:
            data = json.loads(row["data"])
        except (json.JSONDecodeError, TypeError):
            return None

        # In session_message the role lives in the `type` column; legacy `message`
        # rows have no such column and carry the role inside the payload.
        role = _row_value(row, "type") or data.get("role") or data.get("type")

        if role == "user":
            text = str(data.get("text", ""))
        elif role == "assistant":
            # Newer rows carry structured `content` blocks; legacy rows put the
            # whole reply in `text`.
            if "content" in data:
                text = _extract_assistant_text(data.get("content"))
            else:
                text = str(data.get("text", ""))
        elif role == "system":
            text = str(data.get("text", ""))
        else:
            return None

        if not text.strip():
            return None

        return MessageRecord(
            id=row["id"],
            session_id=row["session_id"],
            seq=row["seq"],
            role=role,
            text=text,
            time_created=row["time_created"],
            metadata={k: v for k, v in data.items() if k not in ("content", "text")},
        )


# Subqueries counting only conversational messages, per schema generation.
_COUNT_V2 = """
    (SELECT COUNT(*) FROM session_message sm
     WHERE sm.session_id = session_v2.id
       AND sm.type IN ('user', 'assistant', 'system'))
"""

_COUNT_LEGACY = """
    (SELECT COUNT(*) FROM message m
     WHERE m.session_id = session.id)
"""


def _row_value(row: sqlite3.Row, key: str) -> Any:
    """Read an optional column from a Row without raising on legacy schemas."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def _dedupe(sessions: Any) -> list[SessionRecord]:
    """Drop repeated session IDs, keeping the first occurrence.

    OpenCode mirrors rows into both session tables while migrating, so the same
    session can come back twice from a union query.
    """
    seen: set[str] = set()
    unique: list[SessionRecord] = []

    for session in sessions:
        if session.id not in seen:
            seen.add(session.id)
            unique.append(session)

    return unique


def _extract_assistant_text(content: Any) -> str:
    """Flatten an assistant message's content blocks into plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "")
            if text:
                parts.append(str(text))
        elif block_type == "tool":
            # Keep a short trace of tool use; useful signal, low token cost.
            parts.append(f"[used {block.get('name', 'tool')}]")
        # Reasoning blocks are deliberately dropped: verbose, low signal.

    return "\n".join(parts)


def epoch_ms_to_iso(timestamp_ms: int) -> str:
    """Convert epoch milliseconds to an ISO 8601 string."""
    return datetime.fromtimestamp(timestamp_ms / 1000).isoformat()


__all__ = ["OpenCodeReader", "epoch_ms_to_iso"]
