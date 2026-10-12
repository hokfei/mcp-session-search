"""MCP server exposing OpenCode session history.

Reads OpenCode's database in read-only mode and maintains a local vector index,
so the assistant can recall past sessions without anything being logged by hand.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from session_search.application.search_service import SessionSearchService
from session_search.domain.errors import RepositoryError
from session_search.domain.model import HealthReport, MessageRecord
from session_search.domain.ports import AnswerWriter, Embedder, SessionSource, VectorStore
from session_search.infrastructure.config import settings
from session_search.infrastructure.llm import LocalAnswerWriter, LocalEmbedder
from session_search.infrastructure.persistence.opencode import OpenCodeReader
from session_search.infrastructure.persistence.vector_index import VectorIndex
from session_search.interfaces.mcp.tools import (
    TOOL_DEFINITIONS,
    GetSessionParams,
    IndexSessionsParams,
    ListSessionsParams,
    RebuildIndexParams,
    SearchSessionsParams,
    outcome_to_result,
    record_to_detail,
    record_to_summary,
    stats_to_report,
)

logger = logging.getLogger(__name__)


class SessionSearchServer:
    """Wires the adapters together and dispatches MCP tool calls."""

    def __init__(
        self,
        source: SessionSource,
        store: VectorStore,
        embedder: Embedder,
        writer: AnswerWriter,
    ) -> None:
        self.source = source
        self.store = store
        self.embedder = embedder
        self.writer = writer

        self.service = SessionSearchService(
            source=source, store=store, embedder=embedder, writer=writer
        )

        self.server = Server(
            settings.server_name,
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )

        self._handlers = {
            "search_sessions": self._search,
            "get_session": self._get_session,
            "list_sessions": self._list_sessions,
            "index_sessions": self._index_sessions,
            "rebuild_index": self._rebuild_index,
            "health_check": self._health_check,
        }

    @classmethod
    def from_settings(cls) -> SessionSearchServer:
        """Build a server from configuration, wired to the local filesystem."""
        source = OpenCodeReader(settings.opencode_db_path)
        store = VectorIndex(settings.database_path, settings.embedding_dim)
        embedder = LocalEmbedder()
        writer = LocalAnswerWriter()

        logger.info(
            "Session memory ready | source=%s | index=%s",
            settings.opencode_db_path,
            settings.database_path,
        )

        return cls(source=source, store=store, embedder=embedder, writer=writer)

    # ==================== MCP protocol handlers ====================

    async def _list_tools(self, context: Any = None, params: Any = None) -> ListToolsResult:
        return ListToolsResult(tools=[Tool(**t) for t in TOOL_DEFINITIONS])

    async def _call_tool(self, context: Any = None, params: Any = None) -> CallToolResult:
        """Dispatch a tool call to its handler."""
        name = getattr(params, "name", None)
        arguments = getattr(params, "arguments", None) or {}

        handler = self._handlers.get(name)
        if handler is None:
            return self._error(
                f"Unknown tool: {name}",
                hint=f"Available tools: {', '.join(sorted(self._handlers))}",
            )

        try:
            return await handler(arguments)
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return self._error(f"{type(e).__name__}: {e}")

    # ==================== Tool implementations ====================

    async def _search(self, args: dict[str, Any]) -> CallToolResult:
        params = SearchSessionsParams(**args)

        result = await self.service.search(
            query=params.query,
            top_k=params.top_k,
            threshold=params.threshold,
            directory=params.directory,
            auto_index=settings.auto_index_on_search,
        )

        if not result.is_ok:
            return self._error(str(result.error))

        return self._json(outcome_to_result(result.value))

    async def _get_session(self, args: dict[str, Any]) -> CallToolResult:
        params = GetSessionParams(**args)

        session_result = await self.source.get_session(params.session_id)
        if not session_result.is_ok:
            return self._error(
                str(session_result.error),
                hint="Use list_sessions to find valid session IDs.",
            )

        messages_result = await self.source.get_session_messages(params.session_id)
        if not messages_result.is_ok:
            return self._error(str(messages_result.error))

        messages = list(messages_result.value)

        # Drop trailing turns that fall past the caller's character budget,
        # rather than truncating mid-message and producing broken text.
        budget = params.max_chars
        kept: list[MessageRecord] = []
        used = 0
        for message in messages:
            cost = len(message.text) + len(message.role) + 2
            if used + cost > budget:
                break
            used += cost
            kept.append(message)

        detail = record_to_detail(session_result.value, tuple(kept))
        return self._json(
            {
                **detail.model_dump(exclude_none=True),
                "messages_returned": len(kept),
                "messages_total": len(messages),
                "truncated": len(kept) < len(messages),
            }
        )

    async def _list_sessions(self, args: dict[str, Any]) -> CallToolResult:
        params = ListSessionsParams(**args)

        sessions_result = await self.source.list_sessions(
            limit=params.limit, offset=params.offset, directory=params.directory
        )
        if not sessions_result.is_ok:
            return self._error(str(sessions_result.error))

        count_result = await self.source.count_sessions()
        total = count_result.value if count_result.is_ok else len(sessions_result.value)

        return self._json(
            {
                "sessions": [record_to_summary(s).model_dump() for s in sessions_result.value],
                "total": total,
                "limit": params.limit,
                "offset": params.offset,
            }
        )

    async def _index_sessions(self, args: dict[str, Any]) -> CallToolResult:
        params = IndexSessionsParams(**args)

        if params.mode == "all":
            result = await self.service.index_all(force=False)
        elif params.mode == "force":
            result = await self.service.index_all(force=True)
        else:
            result = await self.service.index_incremental()

        if not result.is_ok:
            return self._error(str(result.error))

        return self._json(stats_to_report(result.value).model_dump())

    async def _rebuild_index(self, args: dict[str, Any]) -> CallToolResult:
        params = RebuildIndexParams(**args)
        if not params.confirm:
            return self._error(
                "rebuild_index requires confirm=true",
                hint="This deletes every indexed embedding and re-embed all sessions.",
            )

        result = await self.service.index_all(force=True)
        if not result.is_ok:
            return self._error(str(result.error))

        report = stats_to_report(result.value)
        report = report.model_copy(update={"message": f"Rebuilt: {report.message}"})
        return self._json(report.model_dump())

    async def _health_check(self, args: dict[str, Any]) -> CallToolResult:
        count_result = await self.source.count_sessions()
        indexed_result = await self.store.count()

        report = HealthReport(
            opencode_db_readable=settings.opencode_db_path.exists(),
            opencode_db_path=str(settings.opencode_db_path),
            total_sessions=count_result.value if count_result.is_ok else 0,
            indexed_sessions=indexed_result.value if indexed_result.is_ok else 0,
            embedding_model=str(settings.model_path),
            embedding_model_loaded=self.embedder.is_loaded,
            answer_synthesis_available=self.writer.is_available,
        )

        return self._json(report.to_dict())

    # ==================== Response helpers ====================

    @staticmethod
    def _json(payload: Any) -> CallToolResult:
        text = (
            payload.model_dump_json(indent=2)
            if hasattr(payload, "model_dump_json")
            else json.dumps(payload, indent=2, default=str)
        )
        return CallToolResult(content=[TextContent(type="text", text=text)])

    @staticmethod
    def _error(message: str, hint: str | None = None) -> CallToolResult:
        payload: dict[str, Any] = {"error": message}
        if hint:
            payload["hint"] = hint
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, indent=2))],
            isError=True,
        )

    # ==================== Lifecycle ====================

    async def run(self) -> None:
        """Serve over stdio until the client disconnects."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main() -> None:
    """Entry point for `python -m session_search`."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # stdout is reserved for the MCP protocol
    )

    try:
        server = SessionSearchServer.from_settings()
    except RepositoryError as e:
        logger.error("Cannot start: %s", e)
        sys.exit(1)

    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
