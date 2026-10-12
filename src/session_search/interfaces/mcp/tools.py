"""MCP tool schemas and response shaping.

Tool descriptions are written for the model that will call them: they state what
the tool is for and when to reach for it, not how it is implemented.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from session_search.domain.model import (
    IndexStats,
    MessageRecord,
    SearchOutcome,
    SessionMatch,
    SessionRecord,
)

# ==================== Input Schemas ====================


class SearchSessionsParams(BaseModel):
    """Search past OpenCode conversations."""

    query: str = Field(
        min_length=1,
        max_length=500,
        description="Natural language question, e.g. 'which database did we choose?'",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "How many sessions to return. Scores decay quickly past rank 1, so 5 is usually plenty."
        ),
    )
    threshold: float = Field(
        default=0.55,
        ge=-1.0,
        le=1.0,
        description=(
            "Minimum relevance score. The default is calibrated so unrelated "
            "questions return nothing; lower it to widen the net."
        ),
    )
    directory: str | None = Field(
        default=None,
        description="Restrict results to one project directory",
    )


class GetSessionParams(BaseModel):
    """Read one session's full transcript."""

    session_id: str = Field(description="Session ID from search or list results")
    max_chars: int = Field(
        default=20000,
        ge=50,
        le=200000,
        description=(
            "Cap on transcript length in characters. Messages are dropped "
            "whole once the budget runs out, never cut mid-sentence."
        ),
    )


class ListSessionsParams(BaseModel):
    """List recent sessions."""

    limit: int = Field(default=20, ge=1, le=100, description="How many to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    directory: str | None = Field(
        default=None,
        description="Restrict to one project directory",
    )


class IndexSessionsParams(BaseModel):
    """Trigger an indexing pass."""

    mode: str = Field(
        default="incremental",
        pattern="^(incremental|all|force)$",
        description=(
            "incremental = only new or changed sessions; "
            "all = every missing session; force = re-embed everything"
        ),
    )


class RebuildIndexParams(BaseModel):
    """Delete the index and rebuild it from scratch."""

    confirm: bool = Field(
        default=False,
        description="Must be true; this is a destructive operation",
    )


# ==================== Response Shapes ====================


class SearchHit(BaseModel):
    """One matching session."""

    session_id: str
    title: str
    directory: str | None
    date: str
    message_count: int
    relevance: float
    preview: str | None


class SearchResult(BaseModel):
    """A complete search response."""

    query: str
    answer: str
    matches: list[SearchHit]
    match_count: int


class SessionDetail(BaseModel):
    """A session and its messages."""

    session_id: str
    title: str
    directory: str | None
    created: str
    updated: str
    message_count: int
    agent: str | None
    model: str | None
    tokens_input: int
    tokens_output: int
    cost: float
    messages: list[dict[str, str]]


class SessionSummary(BaseModel):
    """A session as listed."""

    session_id: str
    title: str
    directory: str | None
    created: str
    updated: str
    message_count: int
    tokens_total: int


class SessionList(BaseModel):
    """A page of sessions."""

    sessions: list[SessionSummary]
    total: int
    limit: int
    offset: int


class IndexReport(BaseModel):
    """Outcome of an indexing pass."""

    indexed: int
    skipped: int
    sessions_found: int
    message: str


# ==================== Converters ====================


def _iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000).isoformat(timespec="seconds")


def match_to_hit(match: SessionMatch) -> SearchHit:
    return SearchHit(
        session_id=match.session_id,
        title=match.display_title,
        directory=match.directory,
        date=match.created_at.strftime("%Y-%m-%d %H:%M"),
        message_count=match.message_count,
        relevance=round(match.score, 3),
        preview=(match.preview or "")[:200] or None,
    )


def outcome_to_result(outcome: SearchOutcome) -> SearchResult:
    return SearchResult(
        query=outcome.query,
        answer=outcome.answer,
        matches=[match_to_hit(m) for m in outcome.matches],
        match_count=len(outcome.matches),
    )


def record_to_summary(session: SessionRecord) -> SessionSummary:
    return SessionSummary(
        session_id=session.id,
        title=session.title or "untitled session",
        directory=session.directory,
        created=_iso(session.time_created),
        updated=_iso(session.time_updated),
        message_count=session.message_count,
        tokens_total=session.tokens_input + session.tokens_output,
    )


def message_to_dict(message: MessageRecord) -> dict[str, str]:
    return {
        "role": message.role,
        "time": _iso(message.time_created),
        "text": message.text,
    }


def record_to_detail(
    session: SessionRecord,
    messages: tuple[MessageRecord, ...],
) -> SessionDetail:
    return SessionDetail(
        session_id=session.id,
        title=session.title or "untitled session",
        directory=session.directory,
        created=_iso(session.time_created),
        updated=_iso(session.time_updated),
        message_count=session.message_count,
        agent=session.agent,
        model=session.model,
        tokens_input=session.tokens_input,
        tokens_output=session.tokens_output,
        cost=round(session.cost, 4),
        messages=[message_to_dict(m) for m in messages],
    )


def stats_to_report(stats: IndexStats) -> IndexReport:
    return IndexReport(
        indexed=stats.indexed,
        skipped=stats.skipped,
        sessions_found=stats.found,
        message=stats.summary,
    )


# ==================== Tool Definitions ====================


def build_tool_definitions() -> list[dict[str, Any]]:
    """Build the MCP tool manifest advertised via tools/list."""
    return [
        {
            "name": "search_sessions",
            "description": (
                "Search your past OpenCode conversations by meaning. Use this "
                "before asking the user to repeat context: it will recall what "
                "was decided, tried, or built in earlier sessions. Indexes any "
                "new sessions automatically, so results are always current."
            ),
            "inputSchema": SearchSessionsParams.model_json_schema(),
        },
        {
            "name": "get_session",
            "description": (
                "Read the full transcript of one past session. Follow a "
                "search_sessions or list_sessions hit with this to see exactly "
                "what was said."
            ),
            "inputSchema": GetSessionParams.model_json_schema(),
        },
        {
            "name": "list_sessions",
            "description": (
                "List recent OpenCode sessions with titles, dates, and message "
                "counts. Use for orientation, or to find a session ID to pass "
                "to get_session."
            ),
            "inputSchema": ListSessionsParams.model_json_schema(),
        },
        {
            "name": "index_sessions",
            "description": (
                "Index sessions into the search index. Rarely needed: "
                "search_sessions indexes automatically. Use 'force' after "
                "changing the embedding model."
            ),
            "inputSchema": IndexSessionsParams.model_json_schema(),
        },
        {
            "name": "rebuild_index",
            "description": (
                "Delete the search index and re-embed every session from "
                "scratch. Destructive; requires confirm=true."
            ),
            "inputSchema": RebuildIndexParams.model_json_schema(),
        },
        {
            "name": "health_check",
            "description": (
                "Report whether session memory is working: whether OpenCode's "
                "database is readable, how many sessions exist, and how many "
                "are indexed."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


TOOL_DEFINITIONS = build_tool_definitions()

__all__ = [
    "TOOL_DEFINITIONS",
    "GetSessionParams",
    "IndexReport",
    "IndexSessionsParams",
    "ListSessionsParams",
    "RebuildIndexParams",
    "SearchSessionsParams",
    "SessionDetail",
    "SessionList",
    "build_tool_definitions",
    "message_to_dict",
    "match_to_hit",
    "outcome_to_result",
    "record_to_detail",
    "record_to_summary",
    "stats_to_report",
]
