"""Domain and system errors for Session Memory."""

from typing import Generic, TypeVar


class DomainError(Exception):
    """Base class for expected domain errors."""

    pass


class SystemError(Exception):
    """Base class for unexpected system errors."""

    pass


# Domain Errors (Expected - handle explicitly)


class SessionNotFoundError(DomainError):
    """Raised when a session is not found."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class SessionAlreadyExistsError(DomainError):
    """Raised when attempting to create a session that already exists."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session already exists: {session_id}")


class InvalidSessionStateError(DomainError):
    """Raised when session is in invalid state for operation."""

    def __init__(self, session_id: str, current_state: str, required_state: str):
        self.session_id = session_id
        self.current_state = current_state
        self.required_state = required_state
        super().__init__(
            f"Session {session_id} is in state '{current_state}', "
            f"but '{required_state}' is required"
        )


class EmbeddingError(DomainError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        super().__init__(f"Embedding error: {message}")


class SummarizationError(DomainError):
    """Raised when session summarization fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        super().__init__(f"Summarization error: {message}")


class SearchError(DomainError):
    """Raised when semantic search fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        super().__init__(f"Search error: {message}")


# System Errors (Unexpected - let bubble up)


class RepositoryError(SystemError):
    """Raised when repository operations fail unexpectedly."""

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        super().__init__(f"Repository error: {message}")


class ConfigurationError(SystemError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")


class ModelLoadError(SystemError):
    """Raised when LLM model fails to load."""

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        super().__init__(f"Model load error: {message}")


# Result Type for Error Handling

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


class Result(Generic[T, E]):
    """Result type for explicit error handling (no exceptions for control flow)."""

    def __init__(self, ok: bool, value: T | None = None, error: E | None = None):
        self._ok = ok
        self._value = value
        self._error = error

    @property
    def is_ok(self) -> bool:
        return self._ok

    @property
    def is_err(self) -> bool:
        return not self._ok

    @property
    def value(self) -> T:
        if not self._ok:
            raise RuntimeError("Cannot access value of error result")
        return self._value

    @property
    def error(self) -> E:
        if self._ok:
            raise RuntimeError("Cannot access error of success result")
        return self._error

    @classmethod
    def ok(cls, value: T) -> "Result[T, E]":
        return cls(ok=True, value=value)

    @classmethod
    def err(cls, error: E) -> "Result[T, E]":
        return cls(ok=False, error=error)

    def map(self, func) -> "Result":
        """Transform the value if ok."""
        if self._ok:
            return Result.ok(func(self._value))
        return Result.err(self._error)

    def flat_map(self, func) -> "Result":
        """Chain operations that return Result."""
        if self._ok:
            return func(self._value)
        return Result.err(self._error)

    def unwrap_or(self, default: T) -> T:
        """Return value or default if error."""
        return self._value if self._ok else default

    def __repr__(self) -> str:
        if self._ok:
            return f"Result.ok({self._value!r})"
        return f"Result.err({self._error!r})"
