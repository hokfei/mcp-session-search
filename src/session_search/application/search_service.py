"""Search orchestration.

Coordinates indexing and querying across session history. Depends only on the
domain ports, so it can be tested with in-memory fakes and never needs to know
whether the backing store is SQLite or an in-memory list.
"""

import logging

from session_search.domain.errors import RepositoryError, Result, SearchError
from session_search.domain.model import (
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

    async def index_session(self, session: SessionRecord) -> Result[None, RepositoryError]:
        """Embed one session's transcript and store its vector."""
        transcript_result = await self.source.get_session_transcript(
            session.id, max_chars=settings.index_max_transcript_chars
        )
        if not transcript_result.is_ok:
            return Result.err(transcript_result.error)

        transcript = transcript_result.value

        # A session with neither a title nor any text carries no signal.
        if not transcript.strip() and not session.title:
            return Result.ok(None)

        embed_result = await self.embedder.embed(self._index_text(session, transcript))
        if not embed_result.is_ok:
            return Result.err(RepositoryError(f"Embed failed: {embed_result.error}"))

        return await self.store.upsert(
            session_id=session.id,
            embedding=embed_result.value.to_bytes(),
            title=session.title,
            directory=session.directory,
            time_created=session.time_created,
            time_updated=session.time_updated,
            message_count=session.message_count,
            preview=transcript[:200].strip() or None,
        )

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
            return Result.ok(IndexStats(indexed=0, skipped=0, found=0))

        already: set[str] = set()
        if not force:
            ids_result = await self.store.get_indexed_ids()
            already = ids_result.value if ids_result.is_ok else set()

        indexed = 0
        skipped = 0

        for session in sessions:
            # Empty sessions carry no signal, and already-indexed ones are
            # untouched unless force was requested.
            if not session.has_content or (not force and session.id in already):
                skipped += 1
                continue

            result = await self.index_session(session)
            if result.is_ok:
                indexed += 1
            else:
                skipped += 1
                logger.warning("Indexing %s failed: %s", session.id, result.error)

        logger.info("Index pass: %s", IndexStats(indexed, skipped, len(sessions)).summary)
        return Result.ok(IndexStats(indexed=indexed, skipped=skipped, found=len(sessions)))

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
        """Pull real transcript text for the strongest matches."""
        parts: list[str] = []

        for match in matches[:3]:
            transcript_result = await self.source.get_session_transcript(
                match.session_id, max_chars=900
            )
            transcript = transcript_result.value if transcript_result.is_ok else ""
            parts.append(
                f"--- {match.display_title} ---\n{transcript or match.preview or '(empty)'}"
            )

        return "\n\n".join(parts)

    @staticmethod
    def _index_text(session: SessionRecord, transcript: str) -> str:
        """Compose the text that represents a session for embedding."""
        parts: list[str] = []
        if session.title:
            parts.append(session.title)
        if transcript:
            parts.append(transcript)
        return "\n".join(parts)


__all__ = ["SessionSearchService"]
