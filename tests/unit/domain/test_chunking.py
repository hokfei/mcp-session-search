"""Tests for transcript chunking.

The driving case: a long session must remain searchable by content that
appears only near its end. Before chunking, indexing truncated to the first
1500 characters, so late content was invisible.
"""

import pytest

from session_search.domain.chunking import (
    DEFAULT_MAX_CHARS,
    TranscriptChunk,
    chunk_transcript,
)
from tests.conftest import make_message


def make_messages(count: int, body: str = "x" * 200, role: str = "user") -> list:
    """Build `count` messages, each roughly `body` characters long."""
    return [make_message(f"m{i}", seq=i, role=role, text=body) for i in range(count)]


class TestChunking:
    def test_short_transcript_is_one_chunk(self):
        chunks = chunk_transcript(make_messages(2), title="Short")

        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_long_transcript_splits(self):
        chunks = chunk_transcript(make_messages(40), title="Long")

        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self):
        chunks = chunk_transcript(make_messages(40))

        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_every_message_appears_somewhere(self):
        messages = [
            make_message(f"m{i}", seq=i, text=f"marker-{i} " + "y" * 150) for i in range(30)
        ]

        chunks = chunk_transcript(messages)
        combined = "\n".join(c.text for c in chunks)

        for i in range(30):
            assert f"marker-{i}" in combined

    def test_content_from_the_end_survives(self):
        """The bug this replaces: late content must not be truncated away."""
        messages = make_messages(40, body="filler " + "z" * 150)
        messages.append(make_message("last", seq=999, text="the needle in the end"))

        chunks = chunk_transcript(messages)

        assert any("the needle in the end" in c.text for c in chunks)

    def test_chunks_respect_the_size_budget(self):
        chunks = chunk_transcript(make_messages(60), max_chars=DEFAULT_MAX_CHARS, title="Budget")

        for chunk in chunks:
            assert len(chunk.text) <= DEFAULT_MAX_CHARS

    def test_messages_are_not_split_midway(self):
        messages = [make_message(f"m{i}", seq=i, text=f"unique{i} " + "w" * 120) for i in range(25)]

        combined = "\n".join(c.text for c in chunk_transcript(messages))

        for i in range(25):
            assert f"unique{i}" in combined

    def test_title_is_repeated_into_every_chunk(self):
        chunks = chunk_transcript(make_messages(40), title="Kubernetes Setup")

        assert len(chunks) > 1
        assert all("Kubernetes Setup" in c.text for c in chunks)

    def test_consecutive_chunks_overlap(self):
        chunks = chunk_transcript(make_messages(40), overlap_chars=200)

        assert len(chunks) > 1
        # Some content must be shared between neighbours, or a fact spanning a
        # boundary becomes unfindable.
        first_lines = set(chunks[0].text.split("\n"))
        second_lines = set(chunks[1].text.split("\n"))
        assert first_lines & second_lines

    def test_empty_input_yields_nothing(self):
        assert chunk_transcript([]) == ()

    def test_blank_messages_are_dropped(self):
        messages = [
            make_message("m1", seq=0, text="real content"),
            make_message("m2", seq=1, text="   "),
            make_message("m3", seq=2, text=""),
        ]

        chunks = chunk_transcript(messages)

        assert len(chunks) == 1
        assert "real content" in chunks[0].text

    def test_title_only_transcript_still_produces_a_chunk(self):
        chunks = chunk_transcript([make_message("m1", text="only this")], title="T")

        assert len(chunks) == 1
        assert chunks[0].text.startswith("T")

    def test_rejects_non_positive_max_chars(self):
        with pytest.raises(ValueError, match="max_chars"):
            chunk_transcript(make_messages(1), max_chars=0)

    def test_rejects_negative_overlap(self):
        with pytest.raises(ValueError, match="overlap_chars"):
            chunk_transcript(make_messages(1), overlap_chars=-1)

    def test_is_deterministic(self):
        messages = make_messages(30)

        assert chunk_transcript(messages) == chunk_transcript(messages)

    def test_chunks_are_indexed_dataclass_values(self):
        chunks = chunk_transcript(make_messages(5))

        assert isinstance(chunks[0], TranscriptChunk)
        assert chunks[0].index == 0


class TestChunkSizing:
    def test_small_budget_forces_more_chunks(self):
        messages = make_messages(40)

        large = len(chunk_transcript(messages, max_chars=4000))
        small = len(chunk_transcript(messages, max_chars=400))

        assert small > large

    def test_tiny_budget_still_produces_content(self):
        chunks = chunk_transcript(make_messages(10), max_chars=250, title="Tiny")

        assert chunks
        assert all(c.text.strip() for c in chunks)
