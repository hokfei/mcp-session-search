"""Splitting long transcripts into embeddable pieces.

bge-small is trained on 512-token passages, so a session is embedded in windows
rather than truncated to its opening. Taking only the first N characters meant
a long-running session was represented entirely by how it started, and anything
discussed later was invisible to search.

This is a pure function over messages: no I/O, no models, fully testable.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from session_search.domain.model import MessageRecord

# Roughly a 512-token window. bge-small degrades on longer input rather than
# summarising it, so the goal is to fit, not to maximise length.
DEFAULT_MAX_CHARS = 1200

# The tail of each chunk is repeated at the head of the next, so a fact that
# straddles a boundary is still findable from either side.
DEFAULT_OVERLAP_CHARS = 200

# A chunk smaller than this carries too little to be worth its own embedding.
_MIN_BUDGET = 200


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """One embeddable piece of a session's transcript."""

    index: int
    text: str


def chunk_transcript(
    messages: Sequence[MessageRecord],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    title: str | None = None,
) -> tuple[TranscriptChunk, ...]:
    """Split messages into overlapping, size-bounded chunks.

    Messages are never split mid-way, so a chunk boundary never produces half a
    sentence. `title` is repeated into every chunk: an isolated window of a
    long session carries no evidence of what it is, and without it a chunk from
    the middle of an 800-message session is nearly unmatchable.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative")

    lines = [f"{m.role}: {m.text.strip()}" for m in messages if m.text.strip()]
    if not lines:
        return ()

    prefix = f"{title}\n" if title else ""
    # Leave room for the repeated title inside every chunk.
    budget = max(_MIN_BUDGET, max_chars - len(prefix))

    return tuple(
        TranscriptChunk(index, prefix + "\n".join(group))
        for index, group in enumerate(_pack(lines, budget, overlap_chars))
    )


def _pack(
    lines: Sequence[str],
    budget: int,
    overlap_chars: int,
) -> list[list[str]]:
    """Greedily fill chunks to the budget, carrying an overlap between them."""
    groups: list[list[str]] = []
    current: list[str] = []
    used = 0

    for line in lines:
        cost = len(line) + 1
        if current and used + cost > budget:
            groups.append(current)
            current = _overlap_tail(current, overlap_chars)
            used = sum(len(part) + 1 for part in current)
        current.append(line)
        used += cost

    if current:
        groups.append(current)

    return groups


def _overlap_tail(lines: Sequence[str], overlap_chars: int) -> list[str]:
    """The trailing lines that fit within the overlap budget."""
    if overlap_chars <= 0:
        return []

    carried: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if used + cost > overlap_chars:
            break
        carried.insert(0, line)
        used += cost
    return carried
