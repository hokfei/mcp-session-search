"""Tests for the vector index.

Covers per-chunk storage and the collapse to one result per session, which is
what lets a session match on any part of itself rather than only its opening.
"""

import sqlite3
import struct
from pathlib import Path

import pytest

from session_search.domain.model import ChunkVector, SessionMatch
from session_search.infrastructure.persistence.vector_index import (
    VectorIndex,
    cosine_similarity,
)


def pack(values: list[float]) -> bytes:
    """Serialize a float vector the same way EmbeddingVector does."""
    return struct.pack(f"{len(values)}f", *values)


def chunk(*vectors: list[float]) -> tuple[ChunkVector, ...]:
    """Build a chunk tuple from vectors."""
    return tuple(ChunkVector(i, pack(v)) for i, v in enumerate(vectors))


async def store(
    index: VectorIndex,
    session_id: str,
    vectors: list[list[float]],
    title: str | None = "T",
    directory: str | None = None,
    time_updated: int = 1,
) -> None:
    """Index one session with the given chunk vectors."""
    await index.upsert_chunks(
        session_id=session_id,
        chunks=chunk(*vectors),
        title=title,
        directory=directory,
        time_created=1,
        time_updated=time_updated,
        message_count=1,
        preview=None,
    )


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert cosine_similarity(pack([1.0, 2.0]), pack([1.0, 2.0])) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity(pack([1.0, 0.0]), pack([0.0, 1.0])) == pytest.approx(0.0)

    def test_scale_does_not_change_score(self):
        assert cosine_similarity(pack([1.0, 1.0]), pack([5.0, 5.0])) == pytest.approx(1.0)

    def test_mismatched_lengths_score_zero(self):
        assert cosine_similarity(pack([1.0, 2.0]), pack([1.0, 2.0, 3.0])) == 0.0

    def test_zero_vector_scores_zero(self):
        assert cosine_similarity(pack([0.0, 0.0]), pack([1.0, 1.0])) == 0.0


class TestStorage:
    async def test_upsert_then_count(self, index_database: Path):
        index = VectorIndex(index_database)

        result = await index.upsert_chunks(
            "s1",
            chunk([1.0, 0.0]),
            "First",
            "/proj",
            1000,
            2000,
            5,
            "preview",
        )

        assert result.is_ok
        assert (await index.count()).value == 1

    async def test_all_chunks_are_stored(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "s1", [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        assert (await index.count_chunks()).value == 3

    async def test_reindexing_replaces_rather_than_appends(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "s1", [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        await store(index, "s1", [[1.0, 0.0]])

        assert (await index.count_chunks()).value == 1
        assert (await index.count()).value == 1

    async def test_index_survives_reopening(self, index_database: Path):
        await VectorIndex(index_database).upsert_chunks(
            "s1", chunk([1.0, 0.0]), "A", None, 1, 1, 0, None
        )

        assert (await VectorIndex(index_database).count()).value == 1

    async def test_creates_parent_directory(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested" / "index.db"

        VectorIndex(nested)

        assert nested.exists()

    async def test_clear_empties_chunks_and_sessions(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "s1", [[1.0, 0.0], [0.0, 1.0]])

        assert (await index.clear()).is_ok
        assert (await index.count()).value == 0
        assert (await index.count_chunks()).value == 0


class TestSearchCollapsesToSessions:
    async def test_matches_on_any_chunk_not_just_the_first(self, index_database: Path):
        """The whole point: late content must be reachable."""
        index = VectorIndex(index_database)
        # First chunk is unrelated; the second is the one the query matches.
        await store(index, "s1", [[0.0, 1.0], [1.0, 0.0]])

        matches = (await index.search(pack([1.0, 0.0]), threshold=-1.0)).value

        assert [m.session_id for m in matches] == ["s1"]

    async def test_a_session_appears_once_however_many_chunks_match(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "s1", [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        matches = (await index.search(pack([1.0, 0.0]), threshold=-1.0)).value

        assert len(matches) == 1

    async def test_score_is_the_best_chunk_not_an_average(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "s1", [[0.0, 1.0], [1.0, 0.0]])

        match = (await index.search(pack([1.0, 0.0]), threshold=-1.0)).value[0]

        assert match.score == pytest.approx(1.0)


class TestSearch:
    async def test_ranks_by_similarity(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "near", [[1.0, 0.0]])
        await store(index, "far", [[0.0, 1.0]])

        matches = (await index.search(pack([1.0, 0.05]), threshold=-1.0)).value

        assert [m.session_id for m in matches] == ["near", "far"]
        assert matches[0].score > matches[1].score

    async def test_applies_threshold(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "only", [[1.0, 0.0]])

        assert len((await index.search(pack([1.0, 0.0]), threshold=0.99)).value) == 1
        assert len((await index.search(pack([0.0, 1.0]), threshold=0.5)).value) == 0

    async def test_respects_top_k(self, index_database: Path):
        index = VectorIndex(index_database)
        for i in range(5):
            await store(index, f"s{i}", [[1.0, 0.0]])

        assert len((await index.search(pack([1.0, 0.0]), threshold=-1.0, top_k=2)).value) == 2

    async def test_top_k_counts_sessions_not_chunks(self, index_database: Path):
        index = VectorIndex(index_database)
        for i in range(4):
            await store(index, f"s{i}", [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        matches = (await index.search(pack([1.0, 0.0]), threshold=-1.0, top_k=2)).value

        assert len(matches) == 2

    async def test_filters_by_directory(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "a", [[1.0, 0.0]], directory="/alpha")
        await store(index, "b", [[1.0, 0.0]], directory="/beta")

        matches = (await index.search(pack([1.0, 0.0]), threshold=-1.0, directory="/beta")).value

        assert [m.session_id for m in matches] == ["b"]

    async def test_returns_domain_matches(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "s1", [[1.0, 0.0]], title="Title", time_updated=2)

        match = (await index.search(pack([1.0, 0.0]), threshold=-1.0)).value[0]

        assert isinstance(match, SessionMatch)
        assert match.message_count == 1

    async def test_empty_index_returns_nothing(self, index_database: Path):
        result = await VectorIndex(index_database).search(pack([1.0, 0.0]))

        assert result.is_ok
        assert result.value == ()


class TestWatermark:
    async def test_empty_index_reports_zero(self, index_database: Path):
        assert (await VectorIndex(index_database).get_last_indexed_time()).value == 0

    async def test_tracks_the_newest(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "s1", [[1.0, 0.0]], time_updated=5000)
        await store(index, "s2", [[1.0, 0.0]], time_updated=9000)

        assert (await index.get_last_indexed_time()).value == 9000

    async def test_lists_every_indexed_id(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "a", [[1.0, 0.0]])
        await store(index, "b", [[1.0, 0.0]])

        assert (await index.get_indexed_ids()).value == {"a", "b"}

    async def test_maps_each_id_to_its_indexed_time(self, index_database: Path):
        index = VectorIndex(index_database)
        await store(index, "a", [[1.0, 0.0]], time_updated=100)
        await store(index, "b", [[1.0, 0.0]], time_updated=200)
        await store(index, "a", [[1.0, 0.0]], time_updated=300)

        assert (await index.get_indexed_times()).value == {"a": 300, "b": 200}


class TestMigration:
    def test_migrates_a_pre_chunking_index(self, tmp_path: Path):
        """An index written by the previous version must not be abandoned."""
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE indexed_sessions (
                session_id TEXT PRIMARY KEY, title TEXT, directory TEXT,
                time_created INTEGER, time_updated INTEGER,
                message_count INTEGER, preview TEXT, embedding BLOB,
                indexed_at INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO indexed_sessions VALUES (?,?,?,?,?,?,?,?,?)",
            ("old", "Legacy", None, 1, 100, 1, None, pack([1.0, 0.0]), 1),
        )
        conn.commit()
        conn.close()

        VectorIndex(db)  # triggers migration

        with sqlite3.connect(db) as check:
            rows = check.execute("SELECT session_id, chunk_index FROM session_chunks").fetchall()

        assert rows == [("old", 0)]

    async def test_migrated_vectors_are_searchable(self, tmp_path: Path):
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE indexed_sessions (
                session_id TEXT PRIMARY KEY, title TEXT, directory TEXT,
                time_created INTEGER, time_updated INTEGER,
                message_count INTEGER, preview TEXT, embedding BLOB,
                indexed_at INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO indexed_sessions VALUES (?,?,?,?,?,?,?,?,?)",
            ("old", "Legacy", None, 1, 100, 1, None, pack([1.0, 0.0]), 1),
        )
        conn.commit()
        conn.close()

        matches = (await VectorIndex(db).search(pack([1.0, 0.0]), threshold=-1.0)).value

        assert [m.session_id for m in matches] == ["old"]
