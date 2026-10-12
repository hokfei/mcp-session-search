"""Tests for value object invariants."""

import numpy as np
import pytest

from session_search.domain.value_objects import EmbeddingVector, SimilarityScore


class TestEmbeddingVector:
    def test_accepts_one_dimensional_float_arrays(self):
        vector = EmbeddingVector(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert vector.dimension == 3

    def test_rejects_two_dimensional_arrays(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            EmbeddingVector(np.zeros((2, 3), dtype=np.float32))

    def test_rejects_empty_arrays(self):
        with pytest.raises(ValueError, match="empty"):
            EmbeddingVector(np.array([], dtype=np.float32))

    def test_upcasts_to_float32(self):
        vector = EmbeddingVector(np.array([1.0, 2.0], dtype=np.float64))
        assert vector.value.dtype == np.float32

    def test_identical_vectors_have_similarity_one(self):
        a = EmbeddingVector(np.array([1.0, 0.0], dtype=np.float32))
        b = EmbeddingVector(np.array([2.0, 0.0], dtype=np.float32))
        assert a.cosine_similarity(b) == pytest.approx(1.0)

    def test_orthogonal_vectors_have_similarity_zero(self):
        a = EmbeddingVector(np.array([1.0, 0.0], dtype=np.float32))
        b = EmbeddingVector(np.array([0.0, 1.0], dtype=np.float32))
        assert a.cosine_similarity(b) == pytest.approx(0.0)

    def test_zero_vector_has_zero_similarity(self):
        a = EmbeddingVector(np.array([0.0, 0.0], dtype=np.float32))
        b = EmbeddingVector(np.array([1.0, 1.0], dtype=np.float32))
        assert a.cosine_similarity(b) == 0.0

    def test_similarity_rejects_mismatched_dimensions(self):
        a = EmbeddingVector(np.zeros(3, dtype=np.float32))
        b = EmbeddingVector(np.zeros(4, dtype=np.float32))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            a.cosine_similarity(b)

    def test_round_trips_through_bytes(self):
        original = EmbeddingVector(np.array([0.5, -0.25, 1.0], dtype=np.float32))
        restored = EmbeddingVector.from_bytes(original.to_bytes(), dimension=3)
        assert restored.value.tolist() == pytest.approx(original.value.tolist())

    def test_from_bytes_rejects_wrong_dimension(self):
        with pytest.raises(ValueError, match="Expected dimension"):
            EmbeddingVector.from_bytes(b"\x00" * 8, dimension=3)

    def test_equality_compares_values(self):
        a = EmbeddingVector(np.array([1.0, 2.0], dtype=np.float32))
        b = EmbeddingVector(np.array([1.0, 2.0], dtype=np.float32))
        assert a == b


class TestSimilarityScore:
    def test_accepts_scores_in_range(self):
        assert SimilarityScore(0.5).value == 0.5

    def test_rejects_scores_above_one(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            SimilarityScore(1.5)

    def test_threshold_check_is_inclusive(self):
        score = SimilarityScore(0.4)
        assert score.meets_threshold(0.4)
        assert not score.meets_threshold(0.5)

    def test_is_orderable(self):
        assert SimilarityScore(0.2) < SimilarityScore(0.8)

    def test_casts_to_float(self):
        assert float(SimilarityScore(0.25)) == 0.25
