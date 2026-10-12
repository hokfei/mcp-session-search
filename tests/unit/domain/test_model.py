"""Tests for the domain model."""

from datetime import datetime

import pytest

from session_search.domain.model import (
    HealthReport,
    IndexStats,
    MessageRecord,
    SearchOutcome,
    SessionMatch,
    SessionRecord,
)
from tests.conftest import make_message, make_session


class TestSessionRecord:
    def test_exposes_timestamps_as_datetimes(self):
        session = make_session(time_created=1_700_000_000_000)
        assert isinstance(session.created_at, datetime)
        assert session.created_at.timestamp() == pytest.approx(1_700_000_000.0)

    def test_has_content_reflects_message_count(self):
        assert make_session(message_count=1).has_content
        assert not make_session(message_count=0).has_content

    def test_is_immutable(self):
        session = make_session()
        with pytest.raises(AttributeError):
            session.title = "changed"  # type: ignore[misc]


class TestMessageRecord:
    def test_carries_role_and_text(self):
        message = make_message(role="assistant", text="an answer")
        assert message.role == "assistant"
        assert message.text == "an answer"


class TestSessionMatch:
    @staticmethod
    def _match(title: str | None) -> SessionMatch:
        return SessionMatch(
            session_id="s1",
            title=title,
            directory=None,
            time_created=0,
            message_count=0,
            preview=None,
            score=0.5,
        )

    def test_display_title_falls_back_when_untitled(self):
        assert self._match(None).display_title == "untitled session"

    def test_display_title_prefers_real_title(self):
        assert self._match("Real title").display_title == "Real title"


class TestIndexStats:
    def test_summary_reports_all_three_counts(self):
        stats = IndexStats(indexed=10, skipped=2, found=12)
        assert "10" in stats.summary
        assert "2" in stats.summary
        assert "12" in stats.summary


class TestSearchOutcome:
    @staticmethod
    def _match() -> SessionMatch:
        return SessionMatch("s1", "t", None, 0, 0, None, 0.9)

    def test_is_empty_when_no_matches(self):
        assert SearchOutcome("q", (), "nothing", False).is_empty

    def test_reports_match_count(self):
        outcome = SearchOutcome("q", (self._match(),), "answer", True)
        assert not outcome.is_empty
        assert len(outcome) == 1


class TestHealthReport:
    @staticmethod
    def _report(opencode_db_readable: bool = True) -> HealthReport:
        return HealthReport(
            opencode_db_readable=opencode_db_readable,
            opencode_db_path="/tmp/opencode.db",
            total_sessions=10,
            indexed_sessions=8,
            embedding_model="model.gguf",
            embedding_model_loaded=True,
            answer_synthesis_available=False,
        )

    def test_status_is_healthy_when_db_readable(self):
        assert self._report().status == "healthy"

    def test_status_is_unhealthy_when_db_missing(self):
        assert self._report(opencode_db_readable=False).status == "unhealthy"

    def test_to_dict_exposes_expected_keys(self):
        keys = self._report().to_dict()
        assert keys["status"] == "healthy"
        assert keys["total_sessions_in_opencode"] == 10
        assert keys["indexed_sessions"] == 8
        assert keys["answer_synthesis_available"] is False


def test_session_record_is_used_by_fixtures():
    """The fixtures build real domain objects, not stand-ins."""
    assert isinstance(make_session(), SessionRecord)
    assert isinstance(make_message(), MessageRecord)
