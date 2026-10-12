"""Calibrate the similarity threshold against the real session corpus.

Scoring granularity changed from one vector per session to one per chunk, so a
threshold calibrated for the old scheme is no longer meaningful. This
remeasures it.

The metric is not "do scores look separated" but "does the right session rank
first". Each probe names the session expected to answer it, so ranking is
measured directly. Irrelevant probes catch the false positives a threshold low
enough to recall everything would let through.
"""

import asyncio

from session_search.application.search_service import SessionSearchService
from session_search.infrastructure.config import settings
from session_search.infrastructure.llm import LocalAnswerWriter, LocalEmbedder
from session_search.infrastructure.persistence.opencode import OpenCodeReader
from session_search.infrastructure.persistence.vector_index import VectorIndex

# Probes with a known answer. `expect` is a word that must appear in the text
# of the chunk that matched - not in the session title. Retrieval surfaces
# passages, so that is what has to be checked; an earlier version matched only
# titles and wrongly reported a failure when the tool had in fact found the
# right passage in a differently-titled session.
PROBES = [
    ("which python version did we install", "Python312"),
    ("what MCP servers did we configure", "brave-search"),
    ("how much do the free opencode models cost", "free"),
    ("docker installation problems on windows", "docker"),
    ("kotlin ios development from windows", "Kotlin"),
    ("moving the skills to the global config folder", "skills"),
    ("what did we decide about the vector index", "sqlite-vec"),
    ("fixing the failing unit tests", "test"),
    ("ollama model not showing in opencode", "ollama"),
    ("the relationship between opencode UI and CLI", "CLI"),
]

# Probes with no plausible answer anywhere in the history. These should not
# clear the threshold at all.
NEGATIVE = [
    "what is the capital of France",
    "how do I bake sourdough bread",
    "recommend a book about marine biology",
    "how to change a bicycle brake cable",
    "population of Reykjavik",
    "best recipe for carbonara",
    "how do I tie a bowline knot",
]


def service() -> SessionSearchService:
    """Wire the real adapters."""
    return SessionSearchService(
        source=OpenCodeReader(settings.opencode_db_path),
        store=VectorIndex(settings.database_path, settings.embedding_dim),
        embedder=LocalEmbedder(),
        writer=LocalAnswerWriter(),
    )


async def probe(svc: SessionSearchService, query: str, top_k: int = 10):
    """Return ranked results for a query with no threshold filtering."""
    result = await svc.search(query, top_k=top_k, threshold=-1.0, auto_index=False)
    if not result.is_ok:
        raise RuntimeError(result.error)
    return result.value.matches


async def main() -> None:
    svc = service()

    stats = await svc.index_all()
    if not stats.is_ok:
        raise SystemExit(f"indexing failed: {stats.error}")
    print(f"index: {stats.value.summary}\n")

    # ---------------- positives ----------------
    top1: list[float] = []
    found_at_1 = 0
    found_anywhere = 0
    print("--- probes with a known answer ---")
    for query, expect in PROBES:
        matches = await probe(svc, query)
        if not matches:
            print(f"  !! no results at all for {query!r}")
            continue

        # The question is whether the expected content was retrieved, so look
        # at the passage that matched rather than the session's name.
        rank = next(
            (
                i + 1
                for i, m in enumerate(matches)
                if expect.lower() in (m.matched_text or "").lower()
            ),
            None,
        )

        score = matches[0].score
        top1.append(score)
        if rank == 1:
            found_at_1 += 1
        if rank is not None:
            found_anywhere += 1

        marker = f"hit at rank {rank}" if rank else "NOT RETRIEVED"
        print(f"  {score:.3f}  {marker:<16} {query}")
        if not rank:
            print(f"            expected {expect!r} in a matched passage")

    # ---------------- negatives ----------------
    negative_best: list[float] = []
    print("\n--- probes with no answer in the corpus ---")
    for query in NEGATIVE:
        matches = await probe(svc, query, top_k=3)
        best = matches[0].score if matches else 0.0
        negative_best.append(best)
        print(f"  {best:.3f}  {query}")

    # ---------------- verdict ----------------
    print("\n=== calibration ===")
    correct_total = len(PROBES)
    print(f"expected passage ranked first: {found_at_1}/{correct_total}")
    print(f"expected passage retrieved at all: {found_anywhere}/{correct_total}")

    if not top1 or not negative_best:
        raise SystemExit("not enough data")

    top1.sort(reverse=True)
    negative_best.sort(reverse=True)

    weakest_positive = min(top1)
    strongest_negative = max(negative_best)

    print(f"\nlowest correct top-1 : {weakest_positive:.3f}")
    print(f"highest false positive: {strongest_negative:.3f}")

    if weakest_positive > strongest_negative:
        midpoint = (weakest_positive + strongest_negative) / 2
        print(
            f"\nClean separation. A threshold anywhere in "
            f"({strongest_negative:.3f}, {weakest_positive:.3f}) keeps every "
            f"correct hit and rejects every false positive."
        )
        print(f"Suggested threshold: {midpoint:.2f}")
    else:
        print(
            "\nOverlap: no threshold both keeps every correct hit and rejects "
            "every false positive."
        )
        print(
            f"  <= {weakest_positive:.3f} keeps all correct hits but admits "
            f"{sum(1 for s in negative_best if s > strongest_negative)} false "
            f"positive(s) at most."
        )

    current = settings.similarity_threshold
    print(f"\ncurrent configured threshold: {current:.2f}")

    if weakest_positive > strongest_negative:
        if current > weakest_positive:
            print("verdict: current threshold is TOO HIGH - it drops correct hits.")
        elif current <= strongest_negative:
            print("verdict: current threshold is TOO LOW - it admits false positives.")
        else:
            print(f"verdict: current threshold sits inside the safe band.")
    else:
        print("verdict: no clean band; see the overlap note above.")


if __name__ == "__main__":
    asyncio.run(main())