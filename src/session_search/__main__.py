"""Command-line entry point.

Serves the MCP protocol on stdio by default, since that is how OpenCode runs
this. A few operational commands are available for use outside a client, where
there is no one to call a tool:

    python -m session_search                 # serve on stdio (the usual case)
    python -m session_search index           # index new or changed sessions
    python -m session_search reindex         # drop the index and rebuild it
    python -m session_search health          # print a health report
    python -m session_search search "query"  # search from the terminal
"""

import argparse
import asyncio
import json
import sys

from session_search.domain.errors import RepositoryError
from session_search.infrastructure.config import settings
from session_search.interfaces.mcp.server import SessionSearchServer, main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session_search",
        description="Semantic search over OpenCode session history.",
    )

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("index", help="index new or changed sessions")
    sub.add_parser("reindex", help="delete the index and rebuild from scratch")
    sub.add_parser("health", help="print a health report")

    search = sub.add_parser("search", help="search sessions from the terminal")
    search.add_argument("query", help="natural language query")
    search.add_argument("-k", "--top-k", type=int, default=5, help="results to return")
    search.add_argument("-d", "--directory", help="restrict to one project directory")
    search.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="minimum relevance score (default: configured threshold)",
    )

    return parser


def build_service() -> SessionSearchServer:
    """Wire the adapters from configuration."""
    return SessionSearchServer.from_settings()


async def run_index(service: SessionSearchServer, force: bool) -> int:
    """Index sessions and report what happened."""
    result = (
        await service.service.index_all(force=True)
        if force
        else await service.service.index_incremental()
    )

    if not result.is_ok:
        print(f"Indexing failed: {result.error}", file=sys.stderr)
        return 1

    print(result.value.summary)
    return 0


async def run_health(service: SessionSearchServer) -> int:
    """Print a health report as JSON."""
    from session_search.domain.model import HealthReport

    count = await service.source.count_sessions()
    indexed = await service.store.count()

    report = HealthReport(
        opencode_db_readable=settings.opencode_db_path.exists(),
        opencode_db_path=str(settings.opencode_db_path),
        total_sessions=count.value if count.is_ok else 0,
        indexed_sessions=indexed.value if indexed.is_ok else 0,
        embedding_model=str(settings.model_path),
        embedding_model_loaded=service.embedder.is_loaded,
        answer_synthesis_available=service.writer.is_available,
    )

    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.status == "healthy" else 1


async def run_search(service: SessionSearchServer, args: argparse.Namespace) -> int:
    """Run a search and print the results."""
    result = await service.service.search(
        query=args.query,
        top_k=args.top_k,
        threshold=args.threshold,
        directory=args.directory,
        auto_index=settings.auto_index_on_search,
    )

    if not result.is_ok:
        print(f"Search failed: {result.error}", file=sys.stderr)
        return 1

    outcome = result.value
    print(outcome.answer)
    print()

    for rank, match in enumerate(outcome.matches, 1):
        when = match.created_at.strftime("%Y-%m-%d %H:%M")
        print(f"{rank}. [{match.score:.3f}] {match.display_title}  ({when})")
        print(f"   {match.session_id}")
        if match.directory:
            print(f"   {match.directory}")

    return 0


async def run_command(args: argparse.Namespace) -> int:
    """Dispatch a non-server command."""
    try:
        service = build_service()
    except RepositoryError as e:
        print(f"Cannot start: {e}", file=sys.stderr)
        return 1

    if args.command == "index":
        return await run_index(service, force=False)
    if args.command == "reindex":
        return await run_index(service, force=True)
    if args.command == "health":
        return await run_health(service)
    if args.command == "search":
        return await run_search(service, args)

    await main()
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Parse arguments and run."""
    args = build_parser().parse_args(argv)
    return asyncio.run(run_command(args))


if __name__ == "__main__":
    sys.exit(cli())
