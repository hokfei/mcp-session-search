"""Value objects with built-in invariants."""

from typing import Self

import numpy as np


class EmbeddingVector:
    """A normalized, fixed-width vector embedding.

    Enforces float32 storage and 1-dimensional shape so that two vectors can
    always be compared. Serialization is raw little-endian float32 bytes.
    """

    __slots__ = ("_value",)

    def __init__(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise ValueError("EmbeddingVector must wrap a numpy array")
        if value.ndim != 1:
            raise ValueError(f"EmbeddingVector must be 1-dimensional, got {value.ndim}D")
        if value.size == 0:
            raise ValueError("EmbeddingVector cannot be empty")

        self._value = value if value.dtype == np.float32 else value.astype(np.float32)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def dimension(self) -> int:
        return int(self._value.shape[0])

    def cosine_similarity(self, other: Self) -> float:
        """Cosine similarity in [-1, 1]; 0.0 if either vector is all zeros."""
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} vs {other.dimension}")
        norm = float(np.linalg.norm(self._value) * np.linalg.norm(other._value))
        if norm == 0.0:
            return 0.0
        return float(np.dot(self._value, other._value) / norm)

    def to_bytes(self) -> bytes:
        """Serialize for storage."""
        return self._value.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, dimension: int | None = None) -> Self:
        """Deserialize, validating width when one is supplied."""
        array = np.frombuffer(data, dtype=np.float32)
        if dimension is not None and array.shape[0] != dimension:
            raise ValueError(f"Expected dimension {dimension}, got {array.shape[0]}")
        return cls(array)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EmbeddingVector):
            return NotImplemented
        return self._value.shape == other._value.shape and bool(
            np.array_equal(self._value, other._value)
        )

    def __hash__(self) -> int:
        return hash(self._value.tobytes())

    def __repr__(self) -> str:
        return f"EmbeddingVector(dimension={self.dimension})"


class SimilarityScore:
    """A cosine similarity score, constrained to [0, 1]."""

    __slots__ = ("_value",)

    def __init__(self, value: float) -> None:
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"SimilarityScore must be in [-1, 1], got {value}")
        self._value = float(value)

    @property
    def value(self) -> float:
        return self._value

    def meets_threshold(self, threshold: float) -> bool:
        return self._value >= threshold

    def __float__(self) -> float:
        return self._value

    def __lt__(self, other: Self) -> bool:
        return self._value < other._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SimilarityScore):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"SimilarityScore({self._value:.3f})"


__all__ = ["EmbeddingVector", "SimilarityScore"]
