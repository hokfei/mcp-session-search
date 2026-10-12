# Session Search MCP

Semantic search across your OpenCode conversation history.

OpenCode already stores every session you have ever had. This server reads that
database, embeds each session, and lets the assistant search it by meaning - so
it can recall what you decided in a past session instead of asking you to repeat
it. Nothing needs to be logged by hand.

## How it works

```
opencode.db ──read──▶ transcript text ──embed──▶ vector ──▶ session_search.db
   (source of truth)     (1500 chars)   (bge-small)      (our index only)

   query ──embed──▶ cosine similarity ──▶ ranked sessions ──▶ answer
```

Everything runs locally. No API keys, no network calls.

## Tools

| Tool | Purpose |
| ---- | ------- |
| `search_sessions` | Search past conversations by meaning. Indexes new sessions automatically. |
| `get_session` | Read one session's full transcript. |
| `list_sessions` | List recent sessions with titles, dates, and message counts. |
| `index_sessions` | Force an indexing pass. Rarely needed. |
| `rebuild_index` | Wipe the index and re-embed everything. Requires `confirm=true`. |
| `health_check` | Report database readability, session counts, and model status. |

`search_sessions` returns both a ranked list of matching sessions and a written
answer. If no chat model is configured the answer degrades to a plain listing of
matches, which is honest about what it is rather than improvising one.

## Setup

Requires Python 3.11+ and a GGUF embedding model.

```bash
# 1. Get an embedding model
mkdir -p ~/models
curl -L -o ~/models/bge-small-en-v1.5-q8_0.gguf \
  https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/main/bge-small-en-v1.5-q8_0.gguf

# 2. Install
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Register it with OpenCode in `~/.opencode/mcp.json`:

```json
"session-search": {
  "command": "C:\\Users\\you\\opencode-tools\\mcp-session-search\\.venv\\Scripts\\python.exe",
  "args": ["-m", "session_search"],
  "env": {
    "SESSION_SEARCH_MODEL_PATH": "C:\\Users\\you\\models\\bge-small-en-v1.5-q8_0.gguf",
    "SESSION_SEARCH_MODEL_N_GPU_LAYERS": "0"
  }
}
```

Verify the wiring end to end:

```bash
python verify_mcp_config.py
```

That reads your `mcp.json`, starts the server with those exact values, and
confirms it reaches the right database and answers a real search.

## Configuration

All settings read from `SESSION_SEARCH_*` environment variables or a `.env`
file. See `.env.example` for the full list. The ones that matter:

| Variable | Default | Notes |
| -------- | ------- | ----- |
| `MODEL_PATH` | `~/models/bge-small-en-v1.5-q8_0.gguf` | Must be an **embedding** model. |
| `OPENCODE_DB_PATH` | `~/.local/share/opencode/opencode.db` | Read-only. |
| `DATABASE_PATH` | `~/.opencode/session_search.db` | Our index only. |
| `SIMILARITY_THRESHOLD` | `0.55` | Calibrated; see ADR 005. |
| `MODEL_N_GPU_LAYERS` | `0` | `0` is fine; the model is 36 MB. |
| `CHAT_MODEL_PATH` | unset | Optional. Enables written answers. |

## Layout

```
src/session_search/
  domain/           model, ports, errors, value objects - no I/O, no deps
  application/      search_service: orchestrates the ports
  infrastructure/   OpenCode reader, vector index, local models, config
  interfaces/mcp/   MCP server and tool schemas
tests/
  unit/             domain, application, infrastructure, interfaces
  contract/         drives a real server process over stdio
docs/adr/           why things are the way they are
```

Dependencies point inwards: `application` and `domain` know nothing about
SQLite, llama.cpp, or MCP. `SessionSearchService` takes the four domain
protocols, which is why the unit tests drive it with in-memory fakes instead of
a database.

## Development

```bash
pytest tests -q          # 162 tests
pytest tests/unit -q     # fast, no subprocesses
ruff check src tests
calibrate.py             # re-measure the similarity threshold
```

Contract tests spawn a real server process and talk JSON-RPC to it over pipes.
They use a synchronous harness (`tests/contract/mcp_client.py`) on purpose:
MCP's async client uses anyio cancel scopes that a pytest fixture cannot exit
cleanly, which deadlocks instead of failing.

## Design notes

- [ADR 001](docs/adr/001-use-python-for-ml-mcp.md) - why Python
- [ADR 002](docs/adr/002-adopt-result-type.md) - `Result<T,E>` over exceptions
- [ADR 003](docs/adr/003-layered-architecture.md) - layering and dependency inversion
- [ADR 004](docs/adr/004-read-from-opencode-database.md) - why we read rather than store
- [ADR 005](docs/adr/005-embedding-model-and-search.md) - embeddings and vector search