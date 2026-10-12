"""Tests for the OpenCode reader against fixture databases."""

from pathlib import Path

import pytest

from session_search.domain.errors import RepositoryError
from session_search.infrastructure.persistence.opencode import (
    OpenCodeReader,
    epoch_ms_to_iso,
)


class TestConstruction:
    def test_missing_database_raises(self, tmp_path: Path):
        with pytest.raises(RepositoryError, match="not found"):
            OpenCodeReader(tmp_path / "absent.db")

    def test_readable_flag_reflects_presence(self, v2_database: Path):
        assert OpenCodeReader(v2_database).is_readable


class TestListSessions:
    async def test_lists_sessions_from_v2_schema(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).list_sessions()

        assert result.is_ok
        assert [s.id for s in result.value] == ["ses_alpha"]

    async def test_counts_only_conversational_messages(self, v2_database: Path):
        """idle and compaction rows are bookkeeping, not conversation."""
        result = await OpenCodeReader(v2_database).list_sessions()

        # The fixture holds 6 rows: 2 user, 2 assistant, 1 idle, 1 compaction.
        assert result.value[0].message_count == 4

    async def test_transcript_drops_turns_with_no_renderable_text(self, v2_database: Path):
        """An assistant turn with only reasoning yields no text to show."""
        result = await OpenCodeReader(v2_database).get_session_messages("ses_alpha")

        assert len(result.value) == 3

    async def test_filters_by_directory(self, mixed_database: Path):
        reader = OpenCodeReader(mixed_database)
        result = await reader.list_sessions(directory="C:/projects/demo")

        assert result.is_ok
        assert all(s.directory == "C:/projects/demo" for s in result.value)

    async def test_returns_most_recent_first(self, mixed_database: Path):
        result = await OpenCodeReader(mixed_database).list_sessions()

        timestamps = [s.time_updated for s in result.value]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_spans_both_schema_generations(self, mixed_database: Path):
        ids = {s.id for s in (await OpenCodeReader(mixed_database).list_sessions()).value}

        assert ids == {"ses_shared", "ses_modern", "ses_old"}

    async def test_orders_by_recency_not_by_table(self, mixed_database: Path):
        """The legacy row must not always sort after the v2 rows."""
        result = await OpenCodeReader(mixed_database).list_sessions()

        assert result.value[0].id == "ses_old"

    async def test_paginates(self, mixed_database: Path):
        reader = OpenCodeReader(mixed_database)
        first = await reader.list_sessions(limit=2, offset=0)
        second = await reader.list_sessions(limit=2, offset=2)

        assert first.is_ok and second.is_ok
        assert len(first.value) == 2
        assert len(second.value) == 1
        assert {s.id for s in first.value}.isdisjoint({s.id for s in second.value})


class TestGetSession:
    async def test_finds_session_in_v2(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).get_session("ses_alpha")

        assert result.is_ok
        assert result.value.title == "Alpha session"

    async def test_finds_session_in_legacy_schema(self, legacy_database: Path):
        result = await OpenCodeReader(legacy_database).get_session("ses_legacy")

        assert result.is_ok
        assert result.value.message_count == 2

    async def test_unknown_session_is_an_error(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).get_session("ses_nope")

        assert result.is_err
        assert "not found" in str(result.error)


class TestGetSessionMessages:
    async def test_returns_messages_in_order(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).get_session_messages("ses_alpha")

        assert [m.seq for m in result.value] == [0, 1, 2]

    async def test_skips_bookkeeping_message_types(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).get_session_messages("ses_alpha")

        assert {m.role for m in result.value} == {"user", "assistant"}

    async def test_reads_user_text_field(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).get_session_messages("ses_alpha")

        assert result.value[0].text == "how do I install docker"

    async def test_flattens_assistant_content_blocks(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).get_session_messages("ses_alpha")
        assistant = next(m for m in result.value if m.role == "assistant")

        assert "winget" in assistant.text
        assert "[used execute]" in assistant.text

    def _assistant_text(self, message) -> str:
        return message.text

    async def test_drops_reasoning_only_messages(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).get_session_messages("ses_alpha")

        assert not any("only thinking" in m.text for m in result.value)

    async def test_reads_legacy_role_from_json(self, legacy_database: Path):
        result = await OpenCodeReader(legacy_database).get_session_messages("ses_legacy")

        assert [m.role for m in result.value] == ["user", "assistant"]
        assert result.value[0].text == "old database question"

    async def test_reads_legacy_prose_from_parts(self, legacy_part_database: Path):
        """Legacy messages carry no text of their own; it lives in `part`."""
        result = await OpenCodeReader(legacy_part_database).get_session_messages("ses_parts")

        assert [m.role for m in result.value] == ["user", "assistant"]
        assert result.value[0].text == "what does the parser do"
        assert result.value[1].text == "it reads the legacy message table"

    async def test_legacy_parts_exclude_reasoning_and_tool_calls(
        self, legacy_part_database: Path
    ):
        """Only `text` parts are prose; the rest is machinery."""
        result = await OpenCodeReader(legacy_part_database).get_session_messages("ses_parts")

        assert not any("thinking about the parser" in m.text for m in result.value)
        assert not any("step-start" in m.text for m in result.value)

    async def test_legacy_session_readable_when_v2_tables_also_exist(
        self, mixed_database: Path
    ):
        """A session only in the legacy table must not be shadowed by v2 tables."""
        result = await OpenCodeReader(mixed_database).get_session_messages("ses_old")

        assert [m.role for m in result.value] == ["user"]
        assert result.value[0].text == "old"

    async def test_respects_limit(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).get_session_messages("ses_alpha", limit=2)

        assert len(result.value) == 2


class TestTranscript:
    async def test_prefixes_each_message_with_its_role(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).get_session_transcript("ses_alpha")

        assert result.value.startswith("user: how do I install docker")

    async def test_respects_max_chars(self, v2_database: Path):
        result = await OpenCodeReader(v2_database).get_session_transcript("ses_alpha", max_chars=25)

        assert len(result.value) <= 25

    async def test_truncates_very_long_messages(self):
        """One enormous message must not crowd out the rest of the transcript."""
        reader = OpenCodeReader(_long_message_db())
        result = await reader.get_session_transcript("ses_long")

        assert len(result.value) < 10_000
        assert "..." in result.value


class TestGetSessionsSince:
    async def test_returns_only_newer_sessions(self, mixed_database: Path):
        result = await OpenCodeReader(mixed_database).get_sessions_since(1_700_000_150_000)

        assert {s.id for s in result.value} == {"ses_old"}

    async def test_returns_oldest_first_for_resumable_indexing(self, mixed_database: Path):
        result = await OpenCodeReader(mixed_database).get_sessions_since(0)

        timestamps = [s.time_updated for s in result.value]
        assert timestamps == sorted(timestamps)

    async def test_deduplicates_overlapping_ids(self, mixed_database: Path):
        result = await OpenCodeReader(mixed_database).get_sessions_since(0)

        ids = [s.id for s in result.value]
        assert len(ids) == len(set(ids))


class TestCountAndDirectories:
    async def test_counts_unique_sessions_across_tables(self, mixed_database: Path):
        """The shared session exists in both tables but is one session."""
        result = await OpenCodeReader(mixed_database).count_sessions()

        assert result.value == 3

    async def test_counts_legacy_sessions(self, legacy_database: Path):
        assert (await OpenCodeReader(legacy_database).count_sessions()).value == 1

    async def test_lists_distinct_directories(self, mixed_database: Path):
        result = await OpenCodeReader(mixed_database).get_directories()

        assert result.value == ("C:/projects/demo",)


class TestReadOnlyGuarantee:
    async def test_cannot_write_to_opencode_database(self, v2_database: Path):
        """The connection is opened in URI read-only mode."""
        import sqlite3

        reader = OpenCodeReader(v2_database)
        with pytest.raises(sqlite3.OperationalError):
            with reader._conn() as conn:
                conn.execute("DELETE FROM session_v2")


class TestEpochConversion:
    def test_converts_milliseconds_to_iso(self):
        result = epoch_ms_to_iso(1_700_000_000_000)

        assert result.startswith("2023-11-14T")


def _long_message_db(tmp_path: Path = None) -> Path:
    """A database holding a single oversized message."""
    import json
    import sqlite3
    import tempfile

    from tests.conftest import make_session

    directory = tmp_path or Path(tempfile.mkdtemp())
    db = directory / "long.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE session_v2 (
            id TEXT PRIMARY KEY, title TEXT, directory TEXT, project_id TEXT,
            version TEXT, agent TEXT, model TEXT, time_created INTEGER,
            time_updated INTEGER, tokens_input INTEGER, tokens_output INTEGER,
            cost REAL
        );
        CREATE TABLE session_message (
            id TEXT PRIMARY KEY, session_id TEXT, seq INTEGER, type TEXT,
            data TEXT, time_created INTEGER
        );
        """
    )

    session = make_session("ses_long", "Long", message_count=1)
    conn.execute(
        """INSERT INTO session_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            session.id,
            session.title,
            session.directory,
            session.project_id,
            session.version,
            session.agent,
            session.model,
            session.time_created,
            session.time_updated,
            session.tokens_input,
            session.tokens_output,
            session.cost,
        ),
    )
    conn.execute(
        """INSERT INTO session_message VALUES (?,?,?,?,?,?)""",
        ("m1", session.id, 0, "user", json.dumps({"text": "x" * 20_000}), 1),
    )
    conn.commit()
    conn.close()
    return db
