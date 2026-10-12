"""Local model adapters: embeddings (required) and answer writing (optional).

Embeddings come from a BERT-family GGUF, which produces a single pooled vector
per input. Generative chat models produce per-token states instead, which are
not usable for similarity search - hence two separate models rather than one.

Both are loaded lazily so that startup stays fast and the server still answers
listing queries even when no chat model is configured.
"""

import logging
import re
from pathlib import Path

import numpy as np
from llama_cpp import Llama

from session_search.domain.errors import EmbeddingError, ModelLoadError, Result
from session_search.domain.model import SessionMatch
from session_search.domain.value_objects import EmbeddingVector
from session_search.infrastructure.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions about a user's past coding sessions, using only "
    "the conversation excerpts you are given.\n"
    "Rules:\n"
    "- Ground every claim in the excerpts. Never invent a version, path, or "
    "decision that is not in them.\n"
    "- If the excerpts do not answer the question, say so plainly.\n"
    "- Be brief. A short paragraph at most."
)

# Hybrid reasoning models can emit a <think> preamble even through the chat
# template. Strip it so it never reaches the caller or eats the budget.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_LEADING_THINK = re.compile(r"^\s*(thinking process:?|thinking:).*", re.S | re.I)


def _strip_thinking(text: str) -> str:
    """Remove any reasoning preamble from a model response."""
    cleaned = _THINK_BLOCK.sub("", text)
    # An unterminated <think> runs to the end of the response.
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return _LEADING_THINK.sub("", cleaned).strip()


class LocalEmbedder:
    """Turns text into a normalized vector using a local GGUF model."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = Path(model_path or settings.model_path)
        self._dimension = settings.embedding_dim
        self._llm: Llama | None = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def is_available(self) -> bool:
        return self._model_path.exists()

    @property
    def llm(self) -> Llama:
        """Load the model on first use."""
        if self._llm is None:
            try:
                self._llm = Llama(
                    model_path=str(self._model_path),
                    n_ctx=settings.model_n_ctx,
                    n_gpu_layers=settings.model_n_gpu_layers,
                    n_threads=settings.model_n_threads,
                    embedding=True,
                    verbose=False,
                )
                logger.info("Loaded embedding model: %s", self._model_path)
            except Exception as e:
                raise ModelLoadError(f"Failed to load {self._model_path.name}: {e}", e)
        return self._llm

    async def embed(self, text: str) -> Result[EmbeddingVector, EmbeddingError]:
        """Generate a pooled, L2-normalized embedding."""
        if not text or not text.strip():
            return Result.err(EmbeddingError("Cannot embed empty text"))

        try:
            # BERT truncates beyond its trained window; do it explicitly so the
            # behaviour is predictable rather than model-dependent.
            max_chars = settings.model_n_ctx * 3
            if len(text) > max_chars:
                text = text[:max_chars]

            response = self.llm.create_embedding(text)
            vector = np.array(response["data"][0]["embedding"], dtype=np.float32)

            if vector.ndim != 1:
                return Result.err(EmbeddingError(f"Unexpected shape {vector.shape}"))

            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm

            self._dimension = int(vector.shape[0])
            return Result.ok(EmbeddingVector(vector))

        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return Result.err(EmbeddingError(f"Failed to embed: {e}", e))


class LocalAnswerWriter:
    """Writes answers from retrieved excerpts, falling back to a plain listing."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = Path(model_path) if model_path else settings.chat_model_path
        self._llm: Llama | None = None

    @property
    def is_available(self) -> bool:
        """Whether a chat model is configured and its file exists."""
        return self._model_path is not None and Path(self._model_path).exists()

    @property
    def llm(self) -> Llama:
        if self._llm is None:
            if not self.is_available:
                raise ModelLoadError("No chat model configured for answer synthesis")
            try:
                self._llm = Llama(
                    model_path=str(self._model_path),
                    n_ctx=settings.chat_model_n_ctx,
                    n_gpu_layers=settings.model_n_gpu_layers,
                    n_threads=settings.model_n_threads,
                    embedding=False,
                    verbose=False,
                )
                logger.info("Loaded chat model: %s", self._model_path)
            except Exception as e:
                raise ModelLoadError(f"Failed to load chat model: {e}", e)
        return self._llm

    async def synthesize(self, query: str, context: str) -> str | None:
        """Write an answer, or None when synthesis is unavailable or failed.

        Uses the chat completion API rather than raw completion: the model's
        chat template is what keeps hybrid reasoning models (Qwen3.5 and later)
        from emitting a <think> preamble that would consume the entire token
        budget before any answer appears.
        """
        if not self.is_available or not context.strip():
            return None

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {query}\n\n"
                            f"Conversation excerpts:\n{context}\n\n"
                            "Answer the question using only these excerpts, "
                            "in a few sentences. If the excerpts do not "
                            "contain the answer, say so plainly."
                        ),
                    },
                ],
                max_tokens=settings.max_summary_tokens,
                temperature=0.2,
            )

            content = response["choices"][0]["message"].get("content") or ""
            return _strip_thinking(content).strip() or None

        except Exception as e:
            # Never fail a search over a synthesis problem.
            logger.warning("Answer synthesis failed, falling back: %s", e)
            return None

    def compose_listing(
        self,
        query: str,
        matches: tuple[SessionMatch, ...],
    ) -> str:
        """Build a plain listing of matches, with no model involved."""
        if not matches:
            return f'No sessions matched "{query}".'

        lines = [f'Sessions matching "{query}":', ""]
        for match in matches:
            date = match.created_at.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  [{date}] {match.display_title} - "
                f"{match.message_count} messages (relevance {match.score:.2f})"
            )
            lines.append(f"    id: {match.session_id}")

        return "\n".join(lines)


__all__ = ["LocalAnswerWriter", "LocalEmbedder"]
