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

# Rough characters per token for English prose. Deliberately pessimistic: a
# chunk full of code or identifiers tokenises worse than ordinary text, and
# overshooting the context window makes synthesis fail outright.
_CHARS_PER_TOKEN = 3.0

# Room for the chat template, the system prompt, and role markers, which are
# added by the model rather than by us.
_TEMPLATE_RESERVE_TOKENS = 500


def _context_char_budget(sessions: int) -> int:
    """Characters of excerpt that fit alongside the answer inside n_ctx.

    Chunking raised each excerpt from ~900 to ~1200 characters, which is enough
    for three of them to overflow the chat model's context window. Without a
    budget, synthesis fails with "requested tokens exceed context window" and
    silently degrades to a listing.
    """
    available = settings.chat_model_n_ctx - settings.max_summary_tokens - _TEMPLATE_RESERVE_TOKENS
    if available <= 0:
        # Degenerate config; give it a token's worth rather than nothing.
        return int(_CHARS_PER_TOKEN)
    return int(available * _CHARS_PER_TOKEN)


def _clip(text: str, limit: int) -> str:
    """Trim text to a limit, cutting on a line boundary where possible."""
    if not text or len(text) <= limit:
        return text

    clipped = text[:limit]
    boundary = clipped.rfind("\n")
    if boundary > limit // 2:
        return clipped[:boundary]

    return clipped.rsplit(" ", 1)[0] if " " in clipped else clipped


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
        # Reconcile first: our database holds a copy of the conversation, so a
        # session removed from OpenCode must not linger here. Archived sessions
        # are excluded from `live_ids` and pruned along with deleted ones.
        pruned = await self.reconcile()

        sessions_result = await self.source.get_sessions_since(
            since_timestamp,
            limit,
            include_archived=settings.include_archived_sessions,
        )
        if not sessions_result.is_ok:
            return Result.err(sessions_result.error)

        sessions = sessions_result.value
        if not sessions:
            return Result.ok(IndexStats(indexed=0, skipped=0, found=0, chunks=0, pruned=pruned))

        indexed_at: dict[str, int] = {}
        if not force:
            times_result = await self.store.get_indexed_times()
            indexed_at = times_result.value if times_result.is_ok else {}

        indexed = 0
        skipped = 0
        chunks_written = 0

        for session in sessions:
            # Empty sessions carry no signal. An indexed session is skipped only
            # while the index is as new as the source: a session that has kept
            # receiving messages must be re-embedded, or its recent work stays
            # unsearchable.
            up_to_date = session.time_updated <= indexed_at.get(session.id, -1)
            if not session.has_content or (not force and up_to_date):
                skipped += 1
                continue

            result = await self.index_session(session)
            if not result.is_ok:
                skipped += 1
                logger.warning("Indexing %s failed: %s", session.id, result.error)
            elif result.value.indexed:
                indexed += 1
                chunks_written += result.value.chunks
            else:
                # Nothing renderable to embed; do not claim it was indexed.
                skipped += 1

        stats = IndexStats(
            indexed=indexed,
            skipped=skipped,
            found=len(sessions),
            chunks=chunks_written,
            pruned=pruned,
        )
        logger.info("Index pass: %s", stats.summary)
        return Result.ok(stats)

    async def reconcile(self) -> int:
        """Drop index entries for sessions that are gone or archived.

        Returns how many were removed. Failures are logged and swallowed: a
        stale index still answers searches, and a reconciliation error should
        not stop indexing.
        """
        live_result = await self.source.get_all_session_ids(
            include_archived=settings.include_archived_sessions
        )
        if not live_result.is_ok:
            logger.warning("Could not list session IDs: %s", live_result.error)
            return 0

        pruned = await self.store.prune_missing(live_result.value)
        if not pruned.is_ok:
            logger.warning("Reconciliation failed: %s", pruned.error)
            return 0

        if pruned.value:
            logger.info("Pruned %d session(s) no longer in OpenCode", pruned.value)
        return pruned.value

    async def forget_session(self, session_id: str) -> Result[None, RepositoryError]:
        """Remove one session from the index immediately."""
        return await self.store.remove(session_id)

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
        chosen = matches[: settings.answer_context_sessions]
        if not chosen:
            return ""

        budget = _context_char_budget(len(chosen))
        per_session = max(300, budget // len(chosen))

        parts: list[str] = []
        for match in chosen:
            excerpt = match.best_excerpt or await self._opening_of(match.session_id)
            parts.append(
                f"--- {match.display_title} ---\n{_clip(excerpt, per_session) or '(empty)'}"
            )

        return "\n\n".join(parts)

    async def _opening_of(self, session_id: str) -> str:
        """A session's first messages, used when no chunk text is stored."""
        result = await self.source.get_session_transcript(session_id, max_chars=900)
        return result.value if result.is_ok else ""


__all__ = ["SessionSearchService"]
