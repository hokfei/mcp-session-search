"""Calibrate the similarity threshold against real session data.

Scores from bge-small cluster tightly (unrelated text sits around 0.45-0.50,
real matches around 0.75-0.85), so the default threshold decides whether search
returns signal or noise. This measures where the gap actually is.
"""

import asyncio

from session_search.application.search_service import SessionSearchService
from session_search.infrastructure.config import settings
from session_search.infrastructure.llm import LocalAnswerWriter, LocalEmbedder
from session_search.infrastructure.persistence.opencode import OpenCodeReader
from session_search.infrastructure.persistence.vector_index import VectorIndex

# Queries with a known answer in the session history.
RELEVANT = [
    "what MCP servers did we install",
    "which python version is installed",
    "how much do the free opencode models cost",
    "what were the docker installation problems",
    "how do I configure kotlin for iOS from windows",
]

# Queries with no plausible answer anywhere in the history.
IRRELEVANT = [
    "what is the capital of France",
    "how do I bake sourdough bread",
    "recommend a good book about marine biology",
    "how to change a bicycle brake cable",
    "what is the population of Reykjavik",
]


async def main() -> None:
    source = OpenCodeReader(settings.opencode_db_path)
    store = VectorIndex(settings.database_path, settings.embedding_dim)
    embedder = LocalEmbedder()
    service = SessionSearchService(
        source=source, store=store, embedder=embedder, writer=LocalAnswerWriter()
    )

    stats = await service.index_all()
    print(f"index: {stats.value.summary if stats.is_ok else stats.error}\n")

    relevant_scores: list[float] = []
    irrelevant_scores: list[float] = []

    for label, queries, bucket in (
        ("RELEVANT", RELEVANT, relevant_scores),
        ("IRRELEVANT", IRRELEVANT, irrelevant_scores),
    ):
        print(f"--- {label} ---")
        for query in queries:
            result = await service.search(query, top_k=10, threshold=-1.0, auto_index=False)
            if not result.is_ok:
                print(f"  {query!r}: FAILED {result.error}")
                continue
            scores = [m.score for m in result.value.matches]
            bucket.extend(scores)
            preview = ", ".join(f"{s:.3f}" for s in scores[:6])
            print(f"  {query}")
            print(f"    top scores: {preview}")
        print()

    if not relevant_scores or not irrelevant_scores:
        print("Not enough data to calibrate.")
        return

    relevant_scores.sort(reverse=True)
    irrelevant_scores.sort(reverse=True)

    worst_relevant = relevant_scores[-1]
    best_irrelevant = irrelevant_scores[0]

    print("=== calibration ===")
    print(f"relevant   n={len(relevant_scores)}  min={worst_relevant:.3f}  "
          f"median={relevant_scores[len(relevant_scores)//2]:.3f}  max={relevant_scores[0]:.3f}")
    print(f"irrelevant n={len(irrelevant_scores)}  max={best_irrelevant:.3f}  "
          f"median={irrelevant_scores[len(irrelevant_scores)//2]:.3f}")
    print()
    if best_irrelevant < worst_relevant:
        midpoint = (worst_relevant + best_irrelevant) / 2
        print(f"Clean separation. Worst relevant ({worst_relevant:.3f}) > "
              f"best irrelevant ({best_irrelevant:.3f}).")
        print(f"Suggested threshold: {midpoint:.2f}")
    else:
        print("No clean separation at top_k=10; scores overlap.")
        print(f"A threshold above {best_irrelevant:.3f} would start dropping real matches.")


if __name__ == "__main__":
    asyncio.run(main())