# ADR 001: Use Python for ML MCP Server

## Status
Accepted (partially superseded)

The language decision stands. Two supporting assumptions did not survive contact
with the problem and are replaced by
[ADR 005](005-embedding-model-and-search.md):

- ~~"Native support for embedding extraction from LLMs"~~ — generative chat
  models emit per-token states, not fixed-width pooled vectors.
- ~~"Use sqlite-vec for vector search"~~ — the `vec0` extension would not load,
  and was unnecessary at this corpus size.

## Context
We need to build an MCP server that provides cross-session semantic memory using local LLM embeddings. The server needs to:
- Generate embeddings for session content
- Perform semantic search over session history
- Generate summaries of conversations
- Run entirely offline with local models

## Decision
Use **Python** with **llama-cpp-python** for the MCP server implementation.

## Rationale

### Python Advantages for ML Workloads
1. **Mature ML Ecosystem**: llama-cpp-python, transformers, sentence-transformers, FAISS, etc.
2. **Model Compatibility**: Direct support for GGUF models via llama-cpp-python
3. **Embedding Generation**: Native support for embedding extraction from LLMs
4. **Vector Search Libraries**: sqlite-vec, FAISS, Chroma, etc. all have Python bindings
5. **Async Support**: Native async/await for MCP server concurrency

### Alternative: TypeScript/Node.js
- Limited GGUF support (requires WASM compilation)
- Fewer embedding model options
- No native FAISS/sqlite-vec bindings
- Would require calling Python subprocesses or HTTP services

### Alternative: Go
- llama.cpp has Go bindings but less mature
- Limited ML ecosystem
- No sqlite-vec equivalent

### Alternative: Rust
- llama.cpp has Rust bindings (llama-cpp-2)
- Growing ML ecosystem but still early
- Steeper learning curve for team

## Consequences

### Positive
- Access to full Python ML ecosystem
- Direct GGUF model loading with llama-cpp-python
- sqlite-vec for efficient KNN search in SQLite
- Pydantic for MCP tool schema validation (like Zod in TypeScript)
- Rich testing ecosystem (pytest, pytest-asyncio)

### Negative
- Python packaging complexity (solved with pyproject.toml + uv/pip)
- Runtime dependency on Python interpreter
- GIL limitations (mitigated by async I/O and llama.cpp releasing GIL during inference)
- Distribution requires Python environment (solved with pipx or Docker)

## Implementation Notes
- Use Python 3.11+ for better async support and performance
- Use `llama-cpp-python` with optional CUDA support via `[gpu]` extra
- Use a sentence-embedding model, not a generative one (see ADR 005)
- Keep vector search behind a port so the backend is replaceable
- Use `pydantic` + `pydantic-settings` for configuration (Zod equivalent)
- Use `pytest` + `pytest-asyncio` for testing

## References
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

---

See also: [ADR 004](004-read-from-opencode-database.md) for where session data
comes from, and [ADR 005](005-embedding-model-and-search.md) for embeddings and
vector search.

## References
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [sqlite-vec](https://github.com/asg017/sqlite-vec)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)