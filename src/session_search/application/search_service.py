"""Search orchestration.

Coordinates indexing and querying across session history. Depends only on the
domain ports, so it can be tested with in-memory fakes and never needs to know
whether the backing store is SQLite or an in-memory list.
"""

import logging

from session_search.domain.chunking import chunk_transcript
from session_search.domain.errors import RepositoryError, Result, SearchError
from session_search.domain.model import (
    ChunkVector,
    IndexStats,
    SearchOutcome,
    SessionMatch,
    SessionRecord,
)
from session_search.domain.ports import AnswerWriter, Embedder, SessionSource, VectorStore
from session_search.infrastructure.config import settings

logger = logging.getLogger(__name__)


class SessionSearchService:
    """Indexes OpenCode session history and answers questions about it."""

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

    # ==================== Indexing ====================

    async def index_session(self, session: SessionRecord) -> Result[IndexStats, RepositoryError]:
        """Embed a session's transcript, one chunk at a time.

        The whole transcript is covered, not just its opening. Truncating meant
        a long-running session was represented entirely by how it started, so
        anything discussed later - which is usually the work worth recalling -
        was unreachable.
        """
        messages_result = await self.source.get_session_messages(session.id)
        if not messages_result.is_ok:
            return Result.err(messages_result.error)

        chunks = chunk_transcript(
            messages_result.value,
            max_chars=settings.index_chunk_chars,
            overlap_chars=settings.index_chunk_overlap_chars,
            title=session.title,
        )

        if not chunks:
            # Nothing renderable to embed.
            return Result.ok(IndexStats(indexed=0, skipped=0, found=0, chunks=0))

        vectors: list[ChunkVector] = []
        for chunk in chunks:
            embed_result = await self.embedder.embed(chunk.text)
            if not embed_result.is_ok:
                return Result.err(RepositoryError(f"Embed failed: {embed_result.error}"))
            vectors.append(ChunkVector(chunk.index, embed_result.value.to_bytes(), chunk.text))

        stored = await self.store.upsert_chunks(
            session_id=session.id,
            chunks=tuple(vectors),
            title=session.title,
            directory=session.directory,
            time_created=session.time_created,
            time_updated=session.time_updated,
            message_count=session.message_count,
            preview=chunks[0].text[:200].strip() or None,
        )

        if not stored.is_ok:
            return stored

        return Result.ok(IndexStats(indexed=1, skipped=0, found=1, chunks=len(vectors)))

    async def index_all(self, force: bool = False) -> Result[IndexStats, RepositoryError]:
        """Index every session OpenCode knows about."""
        if force:
            await self.store.clear()
        return await self._index_pass(since_timestamp=0, limit=1000, force=force)

    async def index_incremental(self) -> Result[IndexStats, RepositoryError]:
        """Index only sessions changed since the last run."""
        last_result = await self.store.get_last_indexed_time()
        since = last_result.value if last_result.is_ok else 0
        return await self._index_pass(since_timestamp=since, limit=200, force=False)

    async def _index_pass(
        self,
        since_timestamp: int,
        limit: int,
        force: bool,
    ) -> Result[IndexStats, RepositoryError]:
        """Run one indexing pass over sessions changed since a timestamp."""
        sessions_result = await self.source.get_sessions_since(since_timestamp, limit)
        if not sessions_result.is_ok:
            return Result.err(sessions_result.error)

        sessions = sessions_result.value
        if not sessions:
            return Result.ok(IndexStats(indexed=0, skipped=0, found=0, chunks=0))

        already: set[str] = set()
        if not force:
            ids_result = await self.store.get_indexed_ids()
            already = ids_result.value if ids_result.is_ok else set()

        indexed = 0
        skipped = 0
        chunks_written = 0

        for session in sessions:
            # Empty sessions carry no signal, and already-indexed ones are
            # untouched unless force was requested.
            if not session.has_content or (not force and session.id in already):
                skipped += 1
                continue

            result = await self.index_session(session)
            if result.is_ok:
                indexed += 1
                chunks_written += result.value.chunks
            else:
                skipped += 1
                logger.warning("Indexing %s failed: %s", session.id, result.error)

        stats = IndexStats(
            indexed=indexed, skipped=skipped, found=len(sessions), chunks=chunks_written
        )
        logger.info("Index pass: %s", stats.summary)
        return Result.ok(stats)

    # ==================== Search ====================

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        directory: str | None = None,
        auto_index: bool = True,
    ) -> Result[SearchOutcome, SearchError]:
        """Search session history semantically, indexing new sessions first."""
        top_k = top_k if top_k is not None else settings.default_top_k
        threshold = settings.similarity_threshold if threshold is None else threshold

        try:
            if auto_index:
                await self._ensure_indexed()

            embed_result = await self.embedder.embed(query)
            if not embed_result.is_ok:
                return Result.err(SearchError(f"Query embed failed: {embed_result.error}"))

            search_result = await self.store.search(
                embed_result.value.to_bytes(),
                top_k=top_k,
                threshold=threshold,
                directory=directory,
            )
            if not search_result.is_ok:
                return Result.err(SearchError(f"Vector search failed: {search_result.error}"))

            matches = search_result.value
            if not matches:
                return Result.ok(
                    SearchOutcome(
                        query=query,
                        matches=(),
                        answer=f'No sessions matched "{query}".',
                        synthesized=False,
                    )
                )

            answer = await self.writer.synthesize(query, await self._gather_context(matches))
            synthesized = answer is not None
            if answer is None:
                answer = self.writer.compose_listing(query, matches)

            return Result.ok(SearchOutcome(query, matches, answer, synthesized))

        except Exception as e:
            logger.exception("Search failed")
            return Result.err(SearchError(f"Search failed: {e}", e))

    async def _ensure_indexed(self) -> None:
        """Bring the index up to date; a full pass on first run, then incremental."""
        count_result = await self.store.count()
        size = count_result.value if count_result.is_ok else 0

        if size == 0:
            logger.info("Index is empty, running a full indexing pass")
            await self.index_all()
            return

        result = await self.index_incremental()
        if not result.is_ok:
            # A stale index still serves searches; log and continue.
            logger.warning("Incremental indexing failed: %s", result.error)

    async def _gather_context(self, matches: tuple[SessionMatch, ...]) -> str:
        """Pull the most relevant text for each matched session.

        A session can match on any part of itself, so the passage that actually
        matched is shown rather than the session's opening. Without this, a hit
        on a long session hands the model a few hundred characters that have
        nothing to do with the question.
        """
        parts: list[str] = []

        for match in matches[:3]:
            excerpt = match.best_excerpt or await self._opening_of(match.session_id)
            parts.append(f"--- {match.display_title} ---\n{excerpt or '(empty)'}")

        return "\n\n".join(parts)

    async def _opening_of(self, session_id: str) -> str:
        """A session's first messages, used when no chunk text is stored."""
        result = await self.source.get_session_transcript(session_id, max_chars=900)
        return result.value if result.is_ok else ""


__all__ = ["SessionSearchService"]
