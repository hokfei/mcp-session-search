"""End-to-end MCP protocol tests.

Each test starts a real server process against a temporary OpenCode database and
a throwaway index. Nothing is mocked, so a break anywhere in the stack - schema
parsing, the SQLite reader, the embedder, the vector index, tool dispatch -
surfaces here rather than in production.
"""

import json
from pathlib import Path

import pytest

from tests.contract.mcp_client import (
    McpStdioClient,
    server_command,
    server_env,
)

SRC_DIR = Path(__file__).parents[2] / "src"

EXPECTED_TOOLS = {
    "search_sessions",
    "get_session",
    "list_sessions",
    "index_sessions",
    "rebuild_index",
    "health_check",
}


@pytest.fixture
def server(tmp_path: Path, v2_database: Path) -> McpStdioClient:
    """A started server, torn down even if a test fails."""
    client = McpStdioClient(
        command=server_command(SRC_DIR),
        env=server_env(
            SRC_DIR,
            session_search_OPENCODE_DB_PATH=str(v2_database),
            session_search_DATABASE_PATH=str(tmp_path / "index.db"),
        ),
    )
    try:
        client.initialize()
        yield client
    finally:
        client.close()


class TestProtocolHandshake:
    def test_advertises_all_tools(self, server: McpStdioClient):
        assert {t["name"] for t in server.list_tools()} == EXPECTED_TOOLS

    def test_tool_schemas_are_json_objects(self, server: McpStdioClient):
        for tool in server.list_tools():
            assert tool["inputSchema"]["type"] == "object", tool["name"]
            json.dumps(tool["inputSchema"])

    def test_every_tool_documents_itself(self, server: McpStdioClient):
        # The model picks tools by description; an empty one is a dead tool.
        for tool in server.list_tools():
            assert tool["description"].strip(), tool["name"]


class TestListSessions:
    def test_returns_sessions_from_the_database(self, server: McpStdioClient):
        body = server.call_tool("list_sessions", {"limit": 10}).json()

        assert body["total"] == 1
        assert body["sessions"][0]["title"] == "Alpha session"

    def test_echoes_pagination_params(self, server: McpStdioClient):
        body = server.call_tool("list_sessions", {"limit": 1}).json()

        assert body["limit"] == 1
        assert body["offset"] == 0

    def test_counts_only_conversational_messages(self, server: McpStdioClient):
        body = server.call_tool("list_sessions").json()

        # idle and compaction rows are bookkeeping, not conversation.
        assert body["sessions"][0]["message_count"] == 4


class TestGetSession:
    def test_returns_only_conversational_messages(self, server: McpStdioClient):
        body = server.call_tool("get_session", {"session_id": "ses_alpha"}).json()

        assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
        assert body["messages_total"] == 3
        assert body["truncated"] is False

    def test_carries_session_metadata(self, server: McpStdioClient):
        body = server.call_tool("get_session", {"session_id": "ses_alpha"}).json()

        assert body["message_count"] == 4
        assert body["tokens_input"] == 100

    def test_flattens_assistant_content_blocks(self, server: McpStdioClient):
        body = server.call_tool("get_session", {"session_id": "ses_alpha"}).json()
        assistant = body["messages"][1]

        assert "winget" in assistant["text"]
        assert "[used execute]" in assistant["text"]

    def test_unknown_session_reports_an_error_with_a_hint(self, server: McpStdioClient):
        result = server.call_tool("get_session", {"session_id": "ses_nope"})

        assert result.is_error
        assert "hint" in result.json()

    def test_truncates_under_a_small_character_budget(self, server: McpStdioClient):
        body = server.call_tool("get_session", {"session_id": "ses_alpha", "max_chars": 50}).json()

        assert body["truncated"] is True
        assert body["messages_returned"] < body["messages_total"]

    def test_keeps_messages_whole_when_dropping(self, server: McpStdioClient):
        body = server.call_tool("get_session", {"session_id": "ses_alpha", "max_chars": 50}).json()

        # Truncation drops whole turns rather than cutting one in half.
        assert all(m["text"] in ("how do I install docker",) for m in body["messages"])


class TestIndexing:
    def test_indexes_sessions(self, server: McpStdioClient):
        body = server.call_tool("index_sessions", {"mode": "all"}).json()

        assert body["sessions_found"] == 1
        assert body["indexed"] == 1

    def test_second_pass_skips_already_indexed(self, server: McpStdioClient):
        server.call_tool("index_sessions", {"mode": "all"})

        body = server.call_tool("index_sessions", {"mode": "incremental"}).json()

        assert body["indexed"] == 0
        assert body["skipped"] == 1

    def test_rebuild_without_confirm_is_refused(self, server: McpStdioClient):
        result = server.call_tool("rebuild_index", {})

        assert result.is_error
        assert "confirm" in result.json()["error"]

    def test_rebuild_with_confirm_succeeds(self, server: McpStdioClient):
        result = server.call_tool("rebuild_index", {"confirm": True})

        assert not result.is_error
        assert "Rebuilt" in result.json()["message"]


class TestSearch:
    def test_finds_a_session_by_meaning(self, server: McpStdioClient):
        server.call_tool("index_sessions", {"mode": "all"})

        body = server.call_tool("search_sessions", {"query": "install docker", "top_k": 1}).json()

        assert body["match_count"] == 1
        assert body["matches"][0]["session_id"] == "ses_alpha"

    def test_returns_an_answer_alongside_matches(self, server: McpStdioClient):
        server.call_tool("index_sessions", {"mode": "all"})

        body = server.call_tool("search_sessions", {"query": "docker install"}).json()

        assert body["answer"].strip()

    def test_indexes_automatically_on_a_cold_search(self, server: McpStdioClient):
        body = server.call_tool("search_sessions", {"query": "docker"}).json()

        assert body["match_count"] >= 1

    def test_high_threshold_filters_everything_out(self, server: McpStdioClient):
        server.call_tool("index_sessions", {"mode": "all"})

        body = server.call_tool("search_sessions", {"query": "docker", "threshold": 0.999}).json()

        assert body["match_count"] == 0
        assert "No sessions matched" in body["answer"]

    def test_unrelated_query_does_not_match(self, server: McpStdioClient):
        server.call_tool("index_sessions", {"mode": "all"})

        body = server.call_tool(
            "search_sessions",
            {"query": "underwater basket weaving", "threshold": 0.9},
        ).json()

        assert body["match_count"] == 0


class TestHealthCheck:
    def test_reports_healthy(self, server: McpStdioClient):
        body = server.call_tool("health_check").json()

        assert body["status"] == "healthy"
        assert body["opencode_db_readable"] is True
        assert body["total_sessions_in_opencode"] == 1

    def test_reports_index_state(self, server: McpStdioClient):
        server.call_tool("index_sessions", {"mode": "all"})

        body = server.call_tool("health_check").json()

        assert body["indexed_sessions"] == 1


class TestErrorHandling:
    def test_unknown_tool_is_an_error(self, server: McpStdioClient):
        result = server.call_tool("does_not_exist")

        assert result.is_error
        assert "Unknown tool" in result.json()["error"]

    def test_invalid_arguments_are_an_error_not_a_crash(self, server: McpStdioClient):
        result = server.call_tool("search_sessions", {"top_k": 3})

        assert result.is_error

    def test_server_survives_a_bad_call(self, server: McpStdioClient):
        server.call_tool("search_sessions", {"top_k": 3})

        assert server.call_tool("health_check").json()["status"] == "healthy"
