# ADR 005: Dedicated Embedding Model, In-Process Cosine Search

## Status
Accepted

Supersedes the embedding and vector-search assumptions in
[ADR 001](001-use-python-for-ml-mcp.md).

## Context
ADR 001 assumed a generative chat model (Qwen2.5-1.5B-Instruct) could produce
embeddings, and that `sqlite-vec` would provide vector search. Both assumptions
failed in practice.

## Decision
Use **bge-small-en-v1.5** (GGUF, 384-dim) for embeddings, and compute cosine
similarity **in pure Python** rather than through a SQLite extension.

## Rationale

### Generative models do not produce usable embeddings
`llm.embed()` on a causal LM returns one vector per token, not a single pooled
vector. Flattening that array yields a length that varies with the input: the
same model returned 3072, 75264, 786432, and 262144 dimensions for different
inputs. No fixed-width vector, so no comparable distance.

`create_embedding()` has the same problem. The fix is a model trained for
sentence embeddings, where pooling is part of the architecture.

### bge-small-en-v1.5
- Fixed **384-dim** pooled output, so vectors are directly comparable.
- 36 MB, loads in well under a second on CPU.
- Genuinely semantic. Measured on this corpus: paraphrases ("python version on
  laptop" vs "what python version is installed") score **0.825**, while
  unrelated text scores **0.468**.

### Pure Python instead of sqlite-vec
`sqlite-vec`'s `vec0` virtual table module failed to load in this environment
(`no such module: vec0`), which would have required building a native extension
and a C toolchain.

The corpus is ~120 sessions x 384 dims. A linear scan over that is a few
milliseconds, well inside what an MCP tool call tolerates, and it removes a
native build dependency entirely. Scoring lives behind the `VectorStore` port, so
swapping in HNSW or sqlite-vec later changes one class and no caller.

## Threshold Calibration
bge-small scores sit high in absolute terms, so the default similarity threshold
was measured rather than guessed, using `calibrate.py` against the real corpus:

| Query kind                       | Top-1 score |
| -------------------------------- | ----------- |
| Has a real answer in the history | 0.60 - 0.85 |
| No plausible answer anywhere    | <= 0.535    |

**Default threshold: 0.55.** At that value every relevant query returned its
correct session while every irrelevant query returned nothing. The tails overlap
below rank 1 - unrelated queries still produce matches around 0.48-0.53 - which
is why the threshold filters them out rather than ranking them low.

## Consequences

### Positive
- Embeddings are genuinely comparable, so search is semantic rather than lexical.
- No native extension to build, ship, or debug.
- Fast enough on CPU that no GPU build is needed; `MODEL_N_GPU_LAYERS=0`.

### Negative
- Only ~1500 characters of each session are embedded, since bge-small is trained
  on 512-token passages. A long session's later content is not represented.
  Titles and the opening turns carry most of the signal as a result.
- Linear scan does not scale indefinitely. At ~10k sessions it is still
  milliseconds; beyond that the `VectorStore` port needs an ANN implementation.

## References
- [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [ggml-org GGUF build](https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)