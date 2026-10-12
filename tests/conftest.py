"""Shared test fixtures: fake OpenCode databases and in-memory port adapters."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from session_search.domain.errors import EmbeddingError, Result
from session_search.domain.model import MessageRecord, SessionRecord
from session_search.domain.value_objects import EmbeddingVector

# A small fixed vocabulary keeps the fake embedder deterministic and makes
# cosine similarity behave the way real embeddings do: shared words, higher score.
VOCABULARY = (
    "python",
    "docker",
    "kubernetes",
    "database",
    "mcp",
    "model",
    "session",
    "search",
    "embedding",
    "install",
    "config",
    "test",
)


def keyword_vector(text: str, dimension: int | None = None) -> EmbeddingVector:
    """Build a normalized bag-of-words vector over a fixed vocabulary."""
    dim = dimension or len(VOCABULARY)
    vector = np.zeros(dim, dtype=np.float32)
    words = text.lower().replace("-", " ").replace("_", " ").split()

    for word in words:
        if word in VOCABULARY:
            vector[VOCABULARY.index(word)] += 1.0

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm

    return EmbeddingVector(vector)


# ==================== Fake domain ports ====================


class FakeEmbedder:
    """Deterministic stand-in for a real embedding model."""

    def __init__(self, dimension: int = 12, fail: bool = False) -> None:
        self._dimension = dimension
        self._fail = fail
        self._loaded = False
        self.calls: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def embed(self, text: str) -> Result[EmbeddingVector, EmbeddingError]:
        self.calls.append(text)
        if self._fail:
            return Result.err(EmbeddingError("embedder is broken"))
        self._loaded = True
        return Result.ok(keyword_vector(text, self._dimension))


class FakeAnswerWriter:
    """Stands in for a chat model, or reports itself unavailable."""

    def __init__(self, available: bool = True, answer: str | None = None) -> None:
        self._available = available
        self._answer = answer
        self.queries: list[str] = []

    @property
    def is_available(self) -> bool:
        return self._available

    async def synthesize(self, query: str, context: str) -> str | None:
        if not self._available:
            return None
        self.queries.append(query)
        return self._answer if self._answer is not None else f"answered: {query}"

    def compose_listing(self, query: str, matches: tuple) -> str:
        titles = ", ".join(m.display_title for m in matches)
        return f'Sessions matching "{query}": {titles}'


class FakeSource:
    """In-memory session source."""

    def __init__(
        self,
        sessions: tuple[SessionRecord, ...] = (),
        messages: dict[str, tuple[MessageRecord, ...]] | None = None,
    ) -> None:
        self.sessions = sessions
        self._messages = messages or {}

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        directory: str | None = None,
    ) -> Result[tuple[SessionRecord, ...], object]:
        rows = [s for s in self.sessions if directory is None or s.directory == directory]
        rows.sort(key=lambda s: s.time_updated, reverse=True)
        return Result.ok(tuple(rows[offset : offset + limit]))

    async def get_session(self, session_id: str) -> Result[SessionRecord, object]:
        for session in self.sessions:
            if session.id == session_id:
                return Result.ok(session)
        return Result.err(Exception(f"Session not found: {session_id}"))

    async def get_session_messages(
        self, session_id: str, limit: int | None = None
    ) -> Result[tuple[MessageRecord, ...], object]:
        messages = self._messages.get(session_id, ())
        return Result.ok(messages[:limit] if limit else messages)

    async def get_session_transcript(
        self, session_id: str, max_chars: int = 4000
    ) -> Result[str, object]:
        messages = self._messages.get(session_id, ())
        text = "\n".join(f"{m.role}: {m.text}" for m in messages)
        return Result.ok(text[:max_chars])

    async def get_sessions_since(
        self, timestamp_ms: int, limit: int = 500
    ) -> Result[tuple[SessionRecord, ...], object]:
        rows = sorted(
            (s for s in self.sessions if s.time_updated > timestamp_ms),
            key=lambda s: s.time_updated,
        )
        return Result.ok(tuple(rows[:limit]))

    async def count_sessions(self) -> Result[int, object]:
        return Result.ok(len(self.sessions))

    async def get_directories(self) -> Result[tuple[str, ...], object]:
        return Result.ok(tuple(sorted({s.directory for s in self.sessions if s.directory})))


class FakeStore:
    """In-memory vector store backed by the shared cosine helper."""

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

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
    ) -> Result[None, object]:
        from session_search.domain.model import SessionMatch

        # SessionMatch deliberately carries no time_updated; keep it alongside
        # the record because incremental indexing needs it.
        self.records[session_id] = {
            "embedding": embedding,
            "time_updated": time_updated,
            "match": SessionMatch(
                session_id=session_id,
                title=title,
                directory=directory,
                time_created=time_created,
                message_count=message_count,
                preview=preview,
                score=0.0,
            ),
        }
        return Result.ok(None)

    async def search(
        self,
        query_embedding: bytes,
        top_k: int = 5,
        threshold: float = 0.35,
        directory: str | None = None,
    ) -> Result[tuple, object]:
        from dataclasses import replace

        from session_search.infrastructure.persistence.vector_index import cosine_similarity

        scored = []
        for record in self.records.values():
            match = record["match"]
            if directory is not None and match.directory != directory:
                continue
            score = cosine_similarity(query_embedding, record["embedding"])
            if score >= threshold:
                scored.append(replace(match, score=score))

        scored.sort(key=lambda m: m.score, reverse=True)
        return Result.ok(tuple(scored[:top_k]))

    async def get_indexed_ids(self) -> Result[set[str], object]:
        return Result.ok(set(self.records))

    async def get_last_indexed_time(self) -> Result[int, object]:
        if not self.records:
            return Result.ok(0)
        return Result.ok(max(r["time_updated"] for r in self.records.values()))

    async def count(self) -> Result[int, object]:
        return Result.ok(len(self.records))

    async def clear(self) -> Result[None, object]:
        self.records.clear()
        return Result.ok(None)


# ==================== Factory helpers ====================


def make_session(
    session_id: str = "ses_1",
    title: str | None = "Test session",
    directory: str | None = "C:/projects/demo",
    time_created: int = 1_700_000_000_000,
    time_updated: int | None = None,
    message_count: int = 2,
) -> SessionRecord:
    """Build a SessionRecord with sensible defaults."""
    return SessionRecord(
        id=session_id,
        title=title,
        directory=directory,
        project_id="prj_1",
        version="1.0.0",
        agent="build",
        model="test-model",
        time_created=time_created,
        time_updated=time_updated if time_updated is not None else time_created,
        message_count=message_count,
        tokens_input=100,
        tokens_output=200,
        cost=0.0012,
    )


def make_message(
    message_id: str = "msg_1",
    session_id: str = "ses_1",
    seq: int = 0,
    role: str = "user",
    text: str = "hello",
    time_created: int = 1_700_000_000_000,
) -> MessageRecord:
    """Build a MessageRecord with sensible defaults."""
    return MessageRecord(
        id=message_id,
        session_id=session_id,
        seq=seq,
        role=role,
        text=text,
        time_created=time_created,
        metadata={},
    )


# ==================== Fixture databases ====================

_V2_SCHEMA = """
CREATE TABLE session_v2 (
    id TEXT PRIMARY KEY,
    title TEXT,
    directory TEXT,
    project_id TEXT,
    version TEXT,
    agent TEXT,
    model TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost REAL
);
CREATE TABLE session_message (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    seq INTEGER,
    type TEXT,
    data TEXT,
    time_created INTEGER
);
"""

_LEGACY_SCHEMA = """
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    title TEXT,
    directory TEXT,
    project_id TEXT,
    version TEXT,
    agent TEXT,
    model TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost REAL
);
CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    data TEXT,
    time_created INTEGER
);
"""


def _insert_session(conn: sqlite3.Connection, table: str, session: SessionRecord) -> None:
    conn.execute(
        f"""
        INSERT INTO {table}
        (id, title, directory, project_id, version, agent, model,
         time_created, time_updated, tokens_input, tokens_output, cost)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session.id,
            session.title,
            session.directory,
            session.project_id,
            session.version,
            session.agent,
            session.model,
            session.time_created,
            session.time_updated,
            session.tokens_input,
            session.tokens_output,
            session.cost,
        ),
    )


@pytest.fixture
def v2_database(tmp_path: Path) -> Path:
    """A database using only the current session_v2 schema."""
    db = tmp_path / "v2.db"
    conn = sqlite3.connect(db)
    conn.executescript(_V2_SCHEMA)

    session = make_session("ses_alpha", "Alpha session", message_count=3)
    _insert_session(conn, "session_v2", session)

    rows = [
        ("m1", "user", {"text": "how do I install docker"}, 0),
        (
            "m2",
            "assistant",
            {
                "content": [
                    {"type": "reasoning", "text": "thinking hard"},
                    {"type": "text", "text": "Use winget to install docker."},
                    {"type": "tool", "name": "execute"},
                ],
                "cost": 0.01,
            },
            1,
        ),
        ("m3", "user", {"text": "and kubernetes?"}, 2),
        ("m4", "idle", {"time": 5}, 3),
        ("m5", "compaction", {"text": "summary of earlier work"}, 4),
        ("m6", "assistant", {"content": [{"type": "reasoning", "text": "only thinking"}]}, 5),
    ]
    for message_id, msg_type, payload, seq in rows:
        conn.execute(
            """
            INSERT INTO session_message (id, session_id, seq, type, data, time_created)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session.id,
                seq,
                msg_type,
                json.dumps(payload),
                1_700_000_000_000 + seq,
            ),
        )

    conn.commit()
    conn.close()
    return db


@pytest.fixture
def legacy_database(tmp_path: Path) -> Path:
    """A database using only the legacy session/message schema."""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript(_LEGACY_SCHEMA)

    session = make_session("ses_legacy", "Legacy session", message_count=2)
    _insert_session(conn, "session", session)

    rows = [
        ("lm1", {"role": "user", "text": "old database question"}, 0),
        ("lm2", {"role": "assistant", "text": "old database answer"}, 1),
    ]
    for message_id, payload, offset in rows:
        conn.execute(
            "INSERT INTO message (id, session_id, data, time_created) VALUES (?, ?, ?, ?)",
            (message_id, session.id, json.dumps(payload), 1_700_000_000_000 + offset),
        )

    conn.commit()
    conn.close()
    return db


@pytest.fixture
def mixed_database(tmp_path: Path) -> Path:
    """A database holding both schema generations, with one overlapping session."""
    db = tmp_path / "mixed.db"
    conn = sqlite3.connect(db)
    conn.executescript(_V2_SCHEMA + _LEGACY_SCHEMA)

    modern = make_session("ses_shared", "Shared session", message_count=1)
    _insert_session(conn, "session_v2", modern)
    conn.execute(
        """
        INSERT INTO session_message (id, session_id, seq, type, data, time_created)
        VALUES ('sm1', ?, 0, 'user', ?, ?)
        """,
        (modern.id, json.dumps({"text": "shared content"}), 1_700_000_000_000),
    )

    only_modern = make_session("ses_modern", "Modern only", time_created=1_700_000_100_000)
    _insert_session(conn, "session_v2", only_modern)

    # Mirrored row: OpenCode duplicates sessions mid-migration.
    _insert_session(conn, "session", modern)
    only_legacy = make_session("ses_old", "Legacy only", time_created=1_700_000_200_000)
    _insert_session(conn, "session", only_legacy)
    conn.execute(
        "INSERT INTO message (id, session_id, data, time_created) VALUES ('lm1', ?, ?, ?)",
        (only_legacy.id, json.dumps({"role": "user", "text": "old"}), 1_700_000_200_000),
    )

    conn.commit()
    conn.close()
    return db


@pytest.fixture
def index_database(tmp_path: Path) -> Path:
    """A fresh path for the vector index database."""
    return tmp_path / "index.db"
