"""Tests for the vector index."""

import struct
from pathlib import Path

import pytest

from session_search.domain.model import SessionMatch
from session_search.infrastructure.persistence.vector_index import (
    VectorIndex,
    cosine_similarity,
)


def pack(values: list[float]) -> bytes:
    """Serialize a float vector the same way EmbeddingVector does."""
    return struct.pack(f"{len(values)}f", *values)


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


class TestVectorIndex:
    async def test_upsert_then_count(self, index_database: Path):
        index = VectorIndex(index_database)

        result = await index.upsert(
            session_id="s1",
            embedding=pack([1.0, 0.0]),
            title="First",
            directory="/proj",
            time_created=1000,
            time_updated=2000,
            message_count=5,
            preview="preview text",
        )

        assert result.is_ok
        assert (await index.count()).value == 1

    async def test_upsert_replaces_existing_session(self, index_database: Path):
        index = VectorIndex(index_database)

        for title in ("Old title", "New title"):
            await index.upsert(
                session_id="s1",
                embedding=pack([1.0, 0.0]),
                title=title,
                directory=None,
                time_created=1,
                time_updated=1,
                message_count=0,
                preview=None,
            )

        assert (await index.count()).value == 1
        matches = (await index.search(pack([1.0, 0.0]), threshold=-1.0)).value
        assert matches[0].title == "New title"

    async def test_search_ranks_by_similarity(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert("near", pack([1.0, 0.0]), "Near", None, 1, 1, 0, None)
        await index.upsert("far", pack([0.0, 1.0]), "Far", None, 1, 1, 0, None)

        matches = (await index.search(pack([1.0, 0.05]), threshold=-1.0)).value

        assert [m.session_id for m in matches] == ["near", "far"]
        assert matches[0].score > matches[1].score

    async def test_search_applies_threshold(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert("only", pack([1.0, 0.0]), "Only", None, 1, 1, 0, None)

        assert len((await index.search(pack([1.0, 0.0]), threshold=0.99)).value) == 1
        assert len((await index.search(pack([0.0, 1.0]), threshold=0.5)).value) == 0

    async def test_search_respects_top_k(self, index_database: Path):
        index = VectorIndex(index_database)
        for i in range(5):
            await index.upsert(f"s{i}", pack([1.0, 0.0]), f"S{i}", None, 1, 1, 0, None)

        assert len((await index.search(pack([1.0, 0.0]), threshold=-1.0, top_k=2)).value) == 2

    async def test_search_filters_by_directory(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert("a", pack([1.0, 0.0]), "A", "/alpha", 1, 1, 0, None)
        await index.upsert("b", pack([1.0, 0.0]), "B", "/beta", 1, 1, 0, None)

        matches = (await index.search(pack([1.0, 0.0]), threshold=-1.0, directory="/beta")).value

        assert [m.session_id for m in matches] == ["b"]

    async def test_search_returns_domain_matches(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert("s1", pack([1.0, 0.0]), "Title", "/dir", 1000, 2000, 7, "preview")

        match = (await index.search(pack([1.0, 0.0]), threshold=-1.0)).value[0]

        assert isinstance(match, SessionMatch)
        assert match.message_count == 7
        assert match.preview == "preview"

    async def test_search_on_empty_index_returns_nothing(self, index_database: Path):
        result = await VectorIndex(index_database).search(pack([1.0, 0.0]))

        assert result.is_ok
        assert result.value == ()

    async def test_last_indexed_time_tracks_newest(self, index_database: Path):
        index = VectorIndex(index_database)
        assert (await index.get_last_indexed_time()).value == 0

        await index.upsert("s1", pack([1.0, 0.0]), "A", None, 1, 5000, 0, None)
        await index.upsert("s2", pack([1.0, 0.0]), "B", None, 1, 9000, 0, None)

        assert (await index.get_last_indexed_time()).value == 9000

    async def test_get_indexed_ids_lists_everything(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert("a", pack([1.0]), "A", None, 1, 1, 0, None)
        await index.upsert("b", pack([1.0]), "B", None, 1, 1, 0, None)

        assert (await index.get_indexed_ids()).value == {"a", "b"}

    async def test_clear_empties_the_index(self, index_database: Path):
        index = VectorIndex(index_database)
        await index.upsert("a", pack([1.0]), "A", None, 1, 1, 0, None)

        assert (await index.clear()).is_ok
        assert (await index.count()).value == 0

    async def test_index_survives_reopening(self, index_database: Path):
        await VectorIndex(index_database).upsert("a", pack([1.0, 0.0]), "A", None, 1, 1, 0, None)

        assert (await VectorIndex(index_database).count()).value == 1

    async def test_creates_parent_directory(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested" / "index.db"

        VectorIndex(nested)

        assert nested.exists()
