"""Tests for the Result type used in place of exceptions for control flow."""

import pytest

from session_search.domain.errors import RepositoryError, Result


class TestResult:
    def test_ok_result_exposes_value(self):
        result = Result.ok(42)
        assert result.is_ok
        assert not result.is_err
        assert result.value == 42

    def test_err_result_exposes_error(self):
        error = RepositoryError("db is gone")
        result = Result.err(error)
        assert result.is_err
        assert not result.is_ok
        assert result.error is error

    def test_accessing_value_on_error_raises(self):
        result = Result.err(RepositoryError("nope"))
        with pytest.raises(RuntimeError, match="Cannot access value"):
            _ = result.value

    def test_accessing_error_on_success_raises(self):
        result = Result.ok(1)
        with pytest.raises(RuntimeError, match="Cannot access error"):
            _ = result.error

    def test_map_transforms_success_value(self):
        assert Result.ok(2).map(lambda v: v * 3).value == 6

    def test_map_propagates_error_unchanged(self):
        error = RepositoryError("still broken")
        assert Result.err(error).map(lambda v: v * 3).error is error

    def test_flat_map_chains_results(self):
        assert Result.ok(2).flat_map(lambda v: Result.ok(v + 1)).value == 3

    def test_flat_map_short_circuits_on_error(self):
        error = RepositoryError("halt")
        assert Result.err(error).flat_map(lambda v: Result.ok(v)).error is error

    def test_unwrap_or_returns_default_on_error(self):
        assert Result.ok(1).unwrap_or(9) == 1
        assert Result.err(RepositoryError("x")).unwrap_or(9) == 9
