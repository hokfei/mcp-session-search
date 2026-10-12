"""Tests for the answer context budget.

Synthesis fails with "requested tokens exceed context window" if the excerpts
plus the answer do not fit inside the chat model's n_ctx. Chunking raised each
excerpt to ~1200 characters, which was enough to trigger this, and the failure
was silent: the caller just got a plain listing instead of an answer.
"""

import pytest

from session_search.application.search_service import (
    _CHARS_PER_TOKEN,
    _clip,
    _context_char_budget,
)
from session_search.domain.model import SessionMatch
from session_search.infrastructure.config import settings


def make_match(session_id: str = "s1", text: str | None = None) -> SessionMatch:
    """A match whose best excerpt is the given text."""
    return SessionMatch(
        session_id=session_id,
        title=f"Title {session_id}",
        directory=None,
        time_created=0,
        message_count=1,
        preview=None,
        score=0.9,
        matched_text=text,
    )


class TestBudget:
    def test_budget_leaves_room_for_the_answer(self):
        """Excerpts plus answer must fit inside n_ctx. The budget is in characters."""
        budget_tokens = _context_char_budget(1) / _CHARS_PER_TOKEN

        assert budget_tokens + settings.max_summary_tokens < settings.chat_model_n_ctx

    def test_budget_reserves_template_space(self):
        """The template reserve is deliberately not spent on excerpts."""
        full = (settings.chat_model_n_ctx - settings.max_summary_tokens) * _CHARS_PER_TOKEN

        assert _context_char_budget(1) < full

    @pytest.mark.parametrize("sessions", [1, 2, 3, 5, 10])
    def test_budget_scales_with_session_count(self, sessions: int):
        assert _context_char_budget(sessions) > 0

    def test_survives_a_degenerate_configuration(self, monkeypatch):
        monkeypatch.setattr(settings, "chat_model_n_ctx", 100)
        monkeypatch.setattr(settings, "max_summary_tokens", 500)

        assert _context_char_budget(1) > 0

    def test_default_config_fits_three_chunk_excerpts(self):
        """The real failure: three 1200-char chunks did not fit in 2048."""
        three_chunks = 3 * 1200

        assert _context_char_budget(3) >= three_chunks


class TestClip:
    def test_short_text_is_untouched(self):
        assert _clip("hello", 100) == "hello"

    def test_empty_text_is_untouched(self):
        assert _clip("", 100) == ""

    def test_long_text_is_shortened(self):
        assert len(_clip("x" * 5000, 100)) <= 100

    def test_prefers_a_line_boundary(self):
        text = "first line\nsecond line\nthird line\nfourth line"

        clipped = _clip(text, 25)

        assert clipped == "first line\nsecond line"

    def test_falls_back_to_a_word_boundary(self):
        clipped = _clip("alpha bravo charlie delta echo foxtrot", 20)

        assert len(clipped) <= 20
        assert " " not in clipped[-1]

    def test_hard_cuts_when_there_are_no_separators(self):
        assert len(_clip("a" * 500, 50)) <= 50

    def test_never_returns_more_than_the_limit(self):
        for limit in (10, 50, 100, 500, 1000):
            assert len(_clip("word " * 2000, limit)) <= limit


class TestGatherContextFits:
    async def test_gathered_context_respects_the_budget(self):
        from session_search.application.search_service import SessionSearchService
        from tests.conftest import (
            FakeAnswerWriter,
            FakeEmbedder,
            FakeSource,
            FakeStore,
        )

        service = SessionSearchService(
            source=FakeSource(),
            store=FakeStore(),
            embedder=FakeEmbedder(),
            writer=FakeAnswerWriter(),
        )

        matches = tuple(make_match(f"s{i}", "x" * 5000) for i in range(3))
        context = await service._gather_context(matches)

        allowance = _context_char_budget(3) + 200  # section headers
        assert len(context) <= allowance

    async def test_every_session_still_appears(self):
        from session_search.application.search_service import SessionSearchService
        from tests.conftest import (
            FakeAnswerWriter,
            FakeEmbedder,
            FakeSource,
            FakeStore,
        )

        service = SessionSearchService(
            source=FakeSource(),
            store=FakeStore(),
            embedder=FakeEmbedder(),
            writer=FakeAnswerWriter(),
        )

        matches = tuple(make_match(f"s{i}", "y" * 5000) for i in range(3))
        context = await service._gather_context(matches)

        for i in range(3):
            assert f"Title s{i}" in context

    async def test_no_matches_yields_empty_context(self):
        from session_search.application.search_service import SessionSearchService
        from tests.conftest import (
            FakeAnswerWriter,
            FakeEmbedder,
            FakeSource,
            FakeStore,
        )

        service = SessionSearchService(
            source=FakeSource(),
            store=FakeStore(),
            embedder=FakeEmbedder(),
            writer=FakeAnswerWriter(),
        )

        assert await service._gather_context(()) == ""
