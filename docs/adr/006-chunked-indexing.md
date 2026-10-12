# ADR 006: Index Sessions in Chunks, Not as One Vector

## Status
Accepted

Refines the indexing half of [ADR 005](005-embedding-model-and-search.md).

## Context
ADR 005 established bge-small as the embedding model and noted its 512-token
training window as a limitation. The first implementation resolved that by
truncating: a session's transcript was cut to its first 1500 characters and
embedded as a single vector.

That was discovered to be badly wrong, and not by inspection — by using the
tool. Asking it *"why did we abandon sqlite-vec for pure python cosine"* returned
two unrelated sessions. The decision had been made in the conversation running
the tool, and the tool could not find it.

Measured on that session:

| | |
| --- | --- |
| Transcript length | 83,571 characters |
| Characters embedded | 1,500 (**1.8%**) |

The embedded text was the session's opening:

> *"Hello, what MCP can you recommend…"*

The words `sqlite-vec`, `cosine`, and `vec0` appeared nowhere in it.

The failure was worst exactly where it mattered most. Long sessions are the
valuable ones — that is where the accumulated work lives — and truncation made
them the *least* searchable. A session's recent turns, which is what you want to
recall, were the part that got thrown away.

## Decision
Split each session's transcript into overlapping chunks and store one vector
per chunk. Search scores every chunk and collapses to one result per session
using its best-scoring chunk. That chunk's text is stored alongside its vector
and returned with the match.

Chunking rules:

- **1200 characters per chunk.** Fits bge-small's 512-token window.
- **200 characters of overlap** between adjacent chunks, so a fact spanning a
  boundary stays findable from either side.
- **Never split mid-message.** Boundaries fall between messages, so no chunk
  ends mid-sentence.
- **Title repeated into every chunk.** A window from the middle of a long
  session carries no evidence of what it is; without the title such a chunk is
  nearly unmatchable.

## Rationale

### A session should match on any part of itself
Scoring chunks and keeping the maximum means one result per session with the
relevance of its best passage. Callers see sessions, not fragments, and a single
strong passage is enough to rank the whole session highly.

### The matched passage, not the session's opening
Storing chunk text was necessary, not merely convenient. Fixing retrieval alone
was not enough: the session was found, but the answer step still read its first
900 characters, so the model was shown text unrelated to the question and
correctly replied that the excerpts contained nothing relevant. Both halves had
to be fixed.

Verified end to end on the same query:

| Stage | Outcome |
| --- | --- |
| Before | Session absent; two unrelated sessions returned |
| Retrieval fixed | Session found at rank 1, score 0.708 — model still said "no relevant information" |
| Excerpts fixed | Model cites the actual reason from the matched chunk |

### Pure domain logic
Chunking is a function of messages, with no I/O and no model. It lives in
`domain/chunking.py` and is tested in isolation, including the case that drives
the whole change: content appearing only near the *end* of a long session must
remain findable.

### Migrations are handled
An index written before this change stored a single `embedding` column. On open,
that vector is moved into `session_chunks` as chunk 0, so existing sessions stay
searchable rather than silently disappearing. There are tests for both the
migration and the fact that migrated vectors still match.

## Consequences

### Positive
- Long sessions are searchable by anything in them.
- Answers cite the passage that actually matched.
- Storage stays trivial: 117 sessions produced 281 chunks, ~1 MB.

### Negative
- **Indexing is slower.** 281 embeddings instead of 117, because a session now
  costs as many inference calls as it has chunks. A full reindex takes noticeably
  longer than the ~4s it used to.
- **Search scans more vectors.** Still milliseconds at this scale, but the cost
  grows with total chunks rather than session count.
- **Threshold had to be recalibrated.** Scores are now computed per chunk, so
  the value tuned for per-session vectors was meaningless. Remeasured to 0.60;
  the reasoning is in the next section.

## Threshold Recalibration

Scoring is now per chunk rather than per session, so the threshold calibrated
under the old scheme no longer meant what it said and had to be remeasured.
`calibrate.py` was rewritten for the occasion to measure something more useful
than score separation: **whether the expected passage is retrieved, and at what
rank.**

The first version of that script checked probe expectations against session
*titles*, which produced a false failure. Asking about Docker installation
problems was reported as "not found" even though the tool had ranked first the
passage *"try to install docker please, the ones I downloaded do not seem to
work"* - in a session whose title did not contain the word Docker. Retrieval
surfaces passages, so the probe now checks the matched chunk text.

Measured on the local corpus with per-chunk scoring:

| | |
| --- | --- |
| Expected passage ranked first | 8 / 10 |
| Expected passage retrieved at all | 9 / 10 |
| Lowest correct top-1 | 0.641 |
| Highest false positive | 0.564 |

**Default threshold raised from 0.55 to 0.60.** The safe band is
(0.564, 0.641); the old 0.55 sat *below* it and admitted false positives, such
as recommending "how to change a bicycle brake cable" at 0.564.

The one genuine miss was a query about the vector-index decision that discussed
the tool that is running it - the hardest case for retrieval, and a fair one to
miss.

## References
- [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)