"""Tests for deletion and archiving.

This tool keeps its own copy of every conversation, so removing a session from
OpenCode has to remove it here too. Before this, nothing could: `time_archived`
was never read, and the index had no delete path at all. An archived session
stayed fully searchable, and a hard-deleted one left its embeddings and
transcript text behind permanently.
"""

import sqlite3
from pathlib import Path

import pytest

from session_search.infrastructure.persistence.opencode import OpenCodeReader
from session_search.infrastructure.persistence.vector_index import VectorIndex
from tests.conftest import _V2_SCHEMA, make_session


@pytest.fixture
def archived_database(tmp_path: Path) -> Path:
    """A database holding one live and one archived session."""
    db = tmp_path / "archived.db"
    conn = sqlite3.connect(db)
    conn.executescript(_V2_SCHEMA)

    for session in (
        make_session("ses_live", "Live session"),
        make_session("ses_gone", "Archived session", archived=True),
    ):
        conn.execute(
            """
            INSERT INTO session_v2
            (id, title, directory, project_id, version, agent, model,
             time_created, time_updated, tokens_input, tokens_output, cost,
             time_archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                1_700_000_000_000 if session.archived else None,
            ),
        )

    conn.commit()
    conn.close()
    return db


class TestReaderExcludesArchived:
    async def test_list_omits_archived(self, archived_database: Path):
        result = await OpenCodeReader(archived_database).list_sessions()

        assert [s.id for s in result.value] == ["ses_live"]

    async def test_list_can_include_archived(self, archived_database: Path):
        result = await OpenCodeReader(archived_database).list_sessions(include_archived=True)

        assert {s.id for s in result.value} == {"ses_live", "ses_gone"}

    async def test_archived_flag_is_reported(self, archived_database: Path):
        result = await OpenCodeReader(archived_database).list_sessions(include_archived=True)

        flags = {s.id: s.archived for s in result.value}

        assert flags == {"ses_live": False, "ses_gone": True}

    async def test_get_sessions_since_omits_archived(self, archived_database: Path):
        result = await OpenCodeReader(archived_database).get_sessions_since(0)

        assert [s.id for s in result.value] == ["ses_live"]

    async def test_count_omits_archived(self, archived_database: Path):
        assert (await OpenCodeReader(archived_database).count_sessions()).value == 1

    async def test_all_ids_omits_archived(self, archived_database: Path):
        result = await OpenCodeReader(archived_database).get_all_session_ids()

        assert result.value == {"ses_live"}

    async def test_all_ids_can_include_archived(self, archived_database: Path):
        result = await OpenCodeReader(archived_database).get_all_session_ids(include_archived=True)

        assert result.value == {"ses_live", "ses_gone"}

    async def test_get_session_still_finds_an_archived_one(self, archived_database: Path):
        """Hiding a session you already hold the ID for would be surprising."""
        result = await OpenCodeReader(archived_database).get_session("ses_gone")

        assert result.is_ok
        assert result.value.archived is True


class TestStoreDelete:
    async def test_remove_deletes_the_session(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert_chunks("s1", (), "A", None, 1, 1, 1, None)

        assert (await index.remove("s1")).is_ok
        assert (await index.count()).value == 0

    async def test_remove_deletes_its_chunks_too(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert_chunks(
            "s1",
            tuple(_chunk(i) for i in range(3)),
            "A",
            None,
            1,
            1,
            1,
            None,
        )

        await index.remove("s1")

        assert (await index.count_chunks()).value == 0

    async def test_removing_an_unknown_session_is_harmless(self, index_database: Path):
        assert (await VectorIndex(index_database).remove("nope")).is_ok

    async def test_a_removed_session_is_not_searchable(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert_chunks("s1", (_chunk(0, [1.0, 0.0]),), "A", None, 1, 1, 1, None)
        await index.remove("s1")

        assert (await index.search(_pack([1.0, 0.0]), threshold=-1.0)).value == ()


class TestPruneMissing:
    async def test_drops_sessions_that_no_longer_exist(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert_chunks("keep", (_chunk(0),), "Keep", None, 1, 1, 1, None)
        await index.upsert_chunks("gone", (_chunk(0),), "Gone", None, 1, 1, 1, None)

        result = await index.prune_missing({"keep"})

        assert result.value == 1
        assert (await index.get_indexed_ids()).value == {"keep"}

    async def test_drops_their_chunks(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert_chunks(
            "gone", tuple(_chunk(i) for i in range(4)), "G", None, 1, 1, 1, None
        )

        await index.prune_missing(set())

        assert (await index.count_chunks()).value == 0

    async def test_nothing_to_prune_returns_zero(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert_chunks("a", (_chunk(0),), "A", None, 1, 1, 1, None)

        assert (await index.prune_missing({"a"})).value == 0

    async def test_pruning_everything_empties_the_index(self, index_database: Path):
        index = VectorIndex(index_database)
        for name in ("a", "b", "c"):
            await index.upsert_chunks(name, (_chunk(0),), name, None, 1, 1, 1, None)

        assert (await index.prune_missing(set())).value == 3
        assert (await index.count()).value == 0

    async def test_the_index_survives_being_rebuilt_after_a_prune(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert_chunks("a", (_chunk(0),), "A", None, 1, 1, 1, None)
        await index.prune_missing(set())
        await index.upsert_chunks("a", (_chunk(0),), "A", None, 1, 1, 1, None)

        assert (await index.count()).value == 1


def _pack(values: list[float]) -> bytes:
    import struct

    return struct.pack(f"{len(values)}f", *values)


def _chunk(index: int, values: list[float] | None = None):
    from session_search.domain.model import ChunkVector

    return ChunkVector(index, _pack(values or [1.0, 0.0]), f"text {index}")
