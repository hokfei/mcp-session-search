# ADR 002: Adopt Result Type for Error Handling

## Status
Accepted

## Context
We need a consistent error handling strategy across the codebase that:
- Makes errors explicit in function signatures
- Avoids exceptions for control flow
- Works across all layers (domain, application, infrastructure)
- Provides type-safe error handling

## Decision
Adopt a **Result<T, E>** type (similar to Rust's Result, Rust's Result, or functional Either) for all fallible operations.

## Rationale

### Problems with Exceptions for Control Flow
1. **Hidden Control Flow**: Callers don't know what errors to handle
2. **No Compile-Time Safety**: Missing error handling caught at runtime
3. **Exception Overhead**: Stack unwinding cost in hot paths
4. **Inconsistent Handling**: Some code catches, some lets bubble

### Benefits of Result Type
1. **Explicit Errors**: Function signature shows possible errors
2. **Compile-Time Exhaustiveness**: Must handle both Ok and Err
3. **Zero-Cost Abstraction**: No stack unwinding on success path
3. **Composable**: `map`, `flat_map`, `and_then` for chaining
4. **Domain-Error Separation**: Distinguishes expected vs unexpected errors

### Error Taxonomy
```
DomainError (Expected - handle explicitly)
├── SessionNotFoundError
├── EmbeddingError
├── SummarizationError
├── SearchError
└── InvalidSessionStateError

SystemError (Unexpected - let bubble)
├── RepositoryError
├── ConfigurationError
├── ModelLoadError
└── RepositoryError
```

## Implementation

```python
# Generic Result type
T = TypeVar("T")
E = TypeVar("E", bound=Exception)

@dataclass
class Result(Generic[T, E]):
    ok: bool
    value: T | None = None
    error: E | None = None
    
    @classmethod
    def ok(cls, value: T) -> "Result[T, E]": ...
    @classmethod
    def err(cls, error: E) -> "Result[T, E]": ...
    
    def map(self, func) -> "Result": ...
    def flat_map(self, func) -> "Result": ...
    def unwrap_or(self, default: T) -> T: ...
```

### Usage Pattern
```python
# Repository returns Result
async def find_by_id(self, id: SessionId) -> Result[Session, SessionNotFoundError]:
    ...

# Application layer handles explicitly
result = await repo.find_by_id(id)
if not result.ok:
    return Result.err(result.error)  # Propagate domain error
session = result.value  # Safe to access
```

## Consequences

### Positive
- Explicit error handling at all layers
- Domain errors distinguishable from system errors
- Type-safe chaining with `map`/`flat_map`
- No unhandled exceptions in normal flow
- Self-documenting function signatures

### Negative
- More verbose than try/except
- Learning curve for team unfamiliar with Result pattern
- Slight boilerplate for simple cases

## Migration Strategy
1. All new code uses Result type
2. Existing exception-based code wrapped at boundaries
3. Domain layer fully Result-based
4. Application layer propagates Results
5. MCP interface converts Results to tool responses

## References
- [Rust Result Type](https://doc.rust-lang.org/std/result/)
- [Functional Error Handling](https://fsharpforfunandprofit.com/rop/)
- [Pydantic Result Pattern](https://github.com/pydantic/pydantic/discussions/...)