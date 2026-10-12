# ADR 004: Read Sessions from OpenCode's Database

## Status
Accepted

## Context
The first implementation of this server stored its own copy of conversations and
exposed `create_session` / `add_message` tools for the assistant to write to it.

That design does not work. The assistant has no reason to log every turn, and
missing turns make search actively misleading: the tool would return confident
"no results" for work that had actually happened. A memory that depends on the
model remembering to write things down fails precisely when it is most needed.

OpenCode already persists every conversation, with titles, timestamps, project
directories, token counts, and full message content, in its own SQLite database.

## Decision
Read session history directly from OpenCode's database, in **read-only mode**.
Store nothing about sessions ourselves except the derived search index.

OpenCode's path defaults to `~/.local/share/opencode/opencode.db` and is
overridable via `SESSION_SEARCH_OPENCODE_DB_PATH`.

## Rationale

### Zero manual logging
The index is built from data that already exists. There is no failure mode where
a conversation happened but was never recorded.

### Read-only access is enforced, not merely intended
Connections use the SQLite URI form `file:...?mode=ro`, so the server *cannot*
write to OpenCode's data even if a future change tried to. There is a test that
asserts a write attempt raises.

### The database is the source of truth
Anything OpenCode reports about a session - title, cost, token usage, working
directory - is available without duplicating it and going stale.

### Two schema generations are live in the wild
Current OpenCode writes `session_v2` / `session_message`. Older installs use
`session` / `session_message`'s predecessor, `session`. During migration OpenCode
mirrors rows across both. The reader therefore:

- detects which tables exist rather than assuming one layout,
- reads both so historical sessions stay searchable,
- de-duplicates by session ID, since the same session can legitimately appear
  twice mid-migration,
- counts sessions as the size of the ID union, not a sum of row counts.

### Message roles live outside the JSON payload
In `session_message` the role is a column. In legacy `message` rows it is inside
the JSON. Assistant replies differ too: current rows carry structured `content`
blocks, legacy rows put the reply in a plain `text` field. Both shapes are
handled, because getting this wrong silently drops every assistant message from
older sessions.

## Consequences

### Positive
- Works retroactively: all past sessions are searchable immediately.
- No write path, so no risk of corrupting OpenCode's data.
- No write path, so no schema migration burden on our side.

### Negative
- Coupling to an upstream schema we do not control. A breaking change to
  OpenCode's tables would need a reader fix; `health_check` reports the
  database's readability so failures are visible rather than silent.
- Reading OpenCode's database while OpenCode is running depends on SQLite's
  locking, which handles concurrent readers fine.

## Alternatives Considered

### Keep our own store, add a hook
Rejected: requires the assistant to log reliably, which is the exact failure
mode described above.

### Tail OpenCode's log files
Rejected: logs are unstructured, rotated, and carry no session metadata.

### Ask OpenCode to export sessions
Rejected: adds a round trip and a failure mode for no benefit over a local read.

## References
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [SQLite read-only URIs](https://www.sqlite.org/uri.html)