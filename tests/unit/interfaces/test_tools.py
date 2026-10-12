"""Tests for MCP tool schemas and response shaping."""

import json

import pytest
from pydantic import ValidationError

from session_search.domain.model import (
    IndexStats,
    SearchOutcome,
    SessionMatch,
)
from session_search.interfaces.mcp.tools import (
    TOOL_DEFINITIONS,
    GetSessionParams,
    IndexSessionsParams,
    ListSessionsParams,
    RebuildIndexParams,
    SearchSessionsParams,
    match_to_hit,
    message_to_dict,
    outcome_to_result,
    record_to_detail,
    record_to_summary,
    stats_to_report,
)
from tests.conftest import make_message, make_session


def _match(session_id: str = "ses_1", score: float = 0.87) -> SessionMatch:
    return SessionMatch(
        session_id=session_id,
        title="A session",
        directory="/proj",
        time_created=1_700_000_000_000,
        message_count=4,
        preview="some preview",
        score=score,
    )


class TestToolManifest:
    def test_exposes_the_expected_tools(self):
        names = {t["name"] for t in TOOL_DEFINITIONS}

        assert names == {
            "search_sessions",
            "get_session",
            "list_sessions",
            "index_sessions",
            "rebuild_index",
            "health_check",
        }

    def test_every_tool_has_a_description(self):
        for tool in TOOL_DEFINITIONS:
            assert tool["description"].strip(), f"{tool['name']} has no description"

    def test_every_tool_has_an_object_input_schema(self):
        for tool in TOOL_DEFINITIONS:
            assert tool["inputSchema"]["type"] == "object", tool["name"]

    def test_manifest_is_json_serializable(self):
        json.dumps(TOOL_DEFINITIONS)


class TestSearchSessionsParams:
    def test_accepts_a_bare_query(self):
        params = SearchSessionsParams(query="what did we decide?")

        assert params.query == "what did we decide?"
        assert params.top_k == 5

    def test_rejects_an_empty_query(self):
        with pytest.raises(ValidationError):
            SearchSessionsParams(query="")

    def test_rejects_top_k_above_the_ceiling(self):
        with pytest.raises(ValidationError):
            SearchSessionsParams(query="q", top_k=100)

    def test_rejects_out_of_range_threshold(self):
        with pytest.raises(ValidationError):
            SearchSessionsParams(query="q", threshold=5.0)


class TestOtherParamSchemas:
    def test_get_session_requires_an_id(self):
        with pytest.raises(ValidationError):
            GetSessionParams()

    def test_list_sessions_defaults(self):
        params = ListSessionsParams()

        assert params.limit == 20
        assert params.offset == 0

    def test_index_mode_is_constrained(self):
        assert IndexSessionsParams().mode == "incremental"
        with pytest.raises(ValidationError):
            IndexSessionsParams(mode="sideways")

    def test_rebuild_requires_explicit_confirmation(self):
        assert RebuildIndexParams().confirm is False


class TestSearchConverters:
    def test_match_to_hit_rounds_relevance(self):
        assert match_to_hit(_match(score=0.87654)).relevance == 0.877

    def test_match_to_hit_titles_untitled_sessions(self):
        untitled = SessionMatch("s", None, None, 0, 0, None, 0.5)

        assert match_to_hit(untitled).title == "untitled session"

    def test_match_to_hit_truncates_long_previews(self):
        long = SessionMatch("s", "t", None, 0, 0, "x" * 500, 0.5)

        assert len(match_to_hit(long).preview) == 200

    def test_outcome_to_result_counts_matches(self):
        outcome = SearchOutcome("q", (_match(), _match("s2")), "answer", True)

        result = outcome_to_result(outcome)

        assert result.match_count == 2
        assert result.answer == "answer"


class TestIndexReport:
    def test_reports_all_counts(self):
        report = stats_to_report(IndexStats(indexed=3, skipped=1, found=4))

        assert report.indexed == 3
        assert report.skipped == 1
        assert report.sessions_found == 4

    def test_includes_a_human_readable_message(self):
        report = stats_to_report(IndexStats(indexed=3, skipped=1, found=4))

        assert "3" in report.message


class TestSessionShaping:
    def test_message_dict_has_role_time_and_text(self):
        payload = message_to_dict(make_message(role="user", text="hi"))

        assert payload["role"] == "user"
        assert payload["text"] == "hi"
        assert "T" in payload["time"]

    def test_detail_includes_metadata_and_messages(self):
        detail = record_to_detail(
            make_session("ses_x", "Title", message_count=2),
            (make_message("m1", "ses_x", 0, "user", "hello"),),
        )

        assert detail.session_id == "ses_x"
        assert detail.tokens_input == 100
        assert len(detail.messages) == 1

    def test_untitled_session_gets_a_placeholder_title(self):
        summary = record_to_summary(make_session(title=None))

        assert summary.title == "untitled session"

    def test_summary_totals_tokens(self):
        assert record_to_summary(make_session()).tokens_total == 300
