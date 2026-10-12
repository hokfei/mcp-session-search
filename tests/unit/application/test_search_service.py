"""Tests for the search orchestration service, driven through in-memory ports."""

import pytest

from session_search.application.search_service import SessionSearchService
from tests.conftest import (
    FakeAnswerWriter,
    FakeEmbedder,
    FakeSource,
    FakeStore,
    keyword_vector,
    make_message,
    make_session,
)

# Tests pass an explicit low threshold so they exercise search behaviour rather
# than whatever the calibrated production default happens to be.
ANY_MATCH = -1.0


def build_service(
    sessions=(),
    messages=None,
    embedder=None,
    writer=None,
) -> SessionSearchService:
    return SessionSearchService(
        source=FakeSource(sessions, messages),
        store=FakeStore(),
        embedder=embedder or FakeEmbedder(),
        writer=writer or FakeAnswerWriter(),
    )


def service_with_sessions(count: int = 2) -> SessionSearchService:
    sessions = tuple(
        make_session(
            session_id=f"ses_{i}",
            title=f"Session {i}",
            message_count=2,
            time_updated=1_700_000_000_000 + i,
        )
        for i in range(count)
    )
    messages = {
        s.id: (
            make_message(f"{s.id}_m1", s.id, 0, "user", "python install test"),
            make_message(f"{s.id}_m2", s.id, 1, "assistant", "python config test"),
        )
        for s in sessions
    }
    return build_service(sessions, messages)


class TestIndexing:
    async def test_indexes_sessions_with_content(self):
        result = await service_with_sessions(count=2).index_all()

        assert result.is_ok
        assert result.value.indexed == 2
        assert result.value.skipped == 0

    async def test_skips_sessions_without_messages(self):
        service = build_service([make_session("empty", message_count=0)])

        result = await service.index_all()

        assert result.value.indexed == 0
        assert result.value.skipped == 1

    async def test_force_mode_reindexes_everything(self):
        service = service_with_sessions(count=1)
        await service.index_all()

        forced = await service.index_all(force=True)

        assert forced.value.indexed == 1
        assert (await service.store.count()).value == 1

    async def test_second_pass_skips_already_indexed(self):
        service = service_with_sessions(count=2)
        await service.index_all()

        again = await service.index_all()

        assert again.value.indexed == 0
        assert again.value.skipped == 2

    async def test_incremental_indexes_new_sessions_only(self):
        service = service_with_sessions(count=1)
        await service.index_all()

        service.source.sessions += (make_session("ses_new", "New", time_updated=1_800_000_000_000),)
        service.source._messages["ses_new"] = (
            make_message("n1", "ses_new", 0, "user", "docker test"),
        )

        result = await service.index_incremental()

        assert result.value.indexed == 1
        assert (await service.store.count()).value == 2

    async def test_empty_source_reports_nothing_found(self):
        result = await build_service([]).index_all()

        assert result.value.found == 0
        assert result.value.indexed == 0

    async def test_index_text_includes_title_and_transcript(self):
        embedder = FakeEmbedder()
        service = build_service(
            [make_session("s1", "Docker setup")],
            {"s1": (make_message("m1", "s1", 0, "user", "install kubernetes"),)},
            embedder=embedder,
        )

        await service.index_all()

        assert "Docker setup" in embedder.calls[0]
        assert "install kubernetes" in embedder.calls[0]

    async def test_embed_failure_is_counted_as_skipped(self):
        service = build_service(
            [make_session("s1")],
            {"s1": (make_message("m1", "s1", 0, "user", "text"),)},
            embedder=FakeEmbedder(fail=True),
        )

        result = await service.index_all()

        assert result.value.indexed == 0
        assert result.value.skipped == 1


class TestSearch:
    async def test_returns_matching_sessions(self):
        service = service_with_sessions(count=2)
        await service.index_all()

        result = await service.search("python", threshold=ANY_MATCH, auto_index=False)

        assert result.is_ok
        assert len(result.value.matches) == 2

    async def test_ranks_most_similar_first(self):
        service = build_service(
            [
                make_session("python_one", "Python work", message_count=1),
                make_session("docker_one", "Docker work", message_count=1),
            ],
            {
                "python_one": (make_message("m1", "python_one", 0, "user", "python python"),),
                "docker_one": (make_message("m2", "docker_one", 0, "user", "docker"),),
            },
        )
        await service.index_all()

        result = await service.search("python python", threshold=ANY_MATCH, auto_index=False)

        assert result.value.matches[0].session_id == "python_one"

    async def test_respects_top_k(self):
        service = service_with_sessions(count=5)
        await service.index_all()

        result = await service.search("python", top_k=2, threshold=ANY_MATCH, auto_index=False)

        assert len(result.value.matches) == 2

    async def test_filters_by_directory(self):
        sessions = (
            make_session("a", "A", directory="/alpha"),
            make_session("b", "B", directory="/beta"),
        )
        messages = {
            s.id: (make_message(f"m{i}", s.id, 0, "user", "python"),)
            for i, s in enumerate(sessions)
        }
        service = build_service(sessions, messages)
        await service.index_all()

        result = await service.search(
            "python", directory="/beta", threshold=ANY_MATCH, auto_index=False
        )

        assert [m.session_id for m in result.value.matches] == ["b"]

    async def test_synthesized_answer_when_writer_available(self):
        service = build_service(
            [make_session("s1")],
            {"s1": (make_message("m1", "s1", 0, "user", "python test"),)},
            writer=FakeAnswerWriter(available=True, answer="The answer is 3.11."),
        )
        await service.index_all()

        result = await service.search("python", threshold=ANY_MATCH, auto_index=False)

        assert result.value.synthesized is True
        assert result.value.answer == "The answer is 3.11."

    async def test_falls_back_to_listing_without_a_writer(self):
        service = build_service(
            [make_session("s1", "Named session")],
            {"s1": (make_message("m1", "s1", 0, "user", "python test"),)},
            writer=FakeAnswerWriter(available=False),
        )
        await service.index_all()

        result = await service.search("python", threshold=ANY_MATCH, auto_index=False)

        assert result.value.synthesized is False
        assert "Named session" in result.value.answer

    async def test_no_matches_returns_empty_answer(self):
        service = service_with_sessions(count=1)
        await service.index_all()

        result = await service.search("kubernetes", threshold=0.99, auto_index=False)

        assert result.value.is_empty
        assert "No sessions matched" in result.value.answer

    async def test_embed_failure_returns_search_error(self):
        service = build_service(
            [make_session("s1")],
            {"s1": (make_message("m1", "s1", 0, "user", "python"),)},
            embedder=FakeEmbedder(fail=True),
        )

        result = await service.search("python", threshold=ANY_MATCH, auto_index=False)

        assert result.is_err

    async def test_auto_index_populates_an_empty_index(self):
        service = service_with_sessions(count=2)
        assert (await service.store.count()).value == 0

        await service.search("python", threshold=ANY_MATCH, auto_index=True)

        assert (await service.store.count()).value == 2

    async def test_auto_index_disabled_skips_indexing(self):
        service = service_with_sessions(count=2)

        await service.search("python", threshold=ANY_MATCH, auto_index=False)

        assert (await service.store.count()).value == 0


class TestContextGathering:
    async def test_context_includes_real_transcript_text(self):
        service = build_service(
            [make_session("s1", "S1")],
            {"s1": (make_message("m1", "s1", 0, "user", "python install config"),)},
            writer=FakeAnswerWriter(available=True, answer="ok"),
        )
        await service.index_all()

        captured = _capturing_writer(service)

        await service.search("python install", threshold=ANY_MATCH, auto_index=False)

        assert captured, "synthesize should be called when there are matches"
        assert "python install config" in captured[0]

    async def test_context_uses_titles_as_section_headers(self):
        service = build_service(
            [make_session("s1", "Distinctive Title")],
            {"s1": (make_message("m1", "s1", 0, "user", "python install"),)},
        )
        await service.index_all()

        captured = _capturing_writer(service)

        await service.search("python install", threshold=ANY_MATCH, auto_index=False)

        assert "Distinctive Title" in captured[0]


def _capturing_writer(service: SessionSearchService) -> list[str]:
    """Replace the writer with one that records the context it was given."""
    captured: list[str] = []

    class CapturingWriter(FakeAnswerWriter):
        async def synthesize(self, query, context):
            captured.append(context)
            return "ok"

    service.writer = CapturingWriter(available=True)
    return captured


class TestEmbedderIntegration:
    async def test_uses_the_embedder_for_indexing_and_querying(self):
        embedder = FakeEmbedder()
        service = service_with_sessions(count=1)
        service.embedder = embedder

        await service.index_all()
        before = len(embedder.calls)
        await service.search("python", threshold=ANY_MATCH, auto_index=False)

        assert len(embedder.calls) == before + 1

    async def test_vectors_are_comparable_after_indexing(self):
        service = service_with_sessions(count=1)
        await service.index_all()

        stored = next(iter(service.store.records.values()))["embedding"]
        query = keyword_vector("python").to_bytes()

        assert len(stored) == len(query)


class TestDependencyInversion:
    def test_accepts_any_port_implementations(self):
        """The service is constructed from domain protocols, not concrete classes."""
        from session_search.domain.ports import (
            AnswerWriter,
            Embedder,
            SessionSource,
            VectorStore,
        )

        service = build_service()

        assert isinstance(service.source, SessionSource)
        assert isinstance(service.store, VectorStore)
        assert isinstance(service.embedder, Embedder)
        assert isinstance(service.writer, AnswerWriter)


@pytest.mark.parametrize("top_k", [1, 3, 5])
async def test_top_k_variants(top_k: int):
    service = service_with_sessions(count=5)
    await service.index_all()

    result = await service.search("python", top_k=top_k, threshold=ANY_MATCH, auto_index=False)

    assert len(result.value.matches) == top_k
