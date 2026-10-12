# ADR 003: Layered Architecture with Dependency Inversion

## Status
Accepted

## Context
We need an architecture that:
- Separates business logic from infrastructure concerns
- Enables testing without external dependencies
- Allows swapping implementations (SQLite → PostgreSQL, llama.cpp → transformers)
- Follows SOLID principles, especially Dependency Inversion

## Decision
Use **Layered Architecture** with **Dependency Inversion** (Ports and Adapters / Hexagonal Architecture).

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                    INTERFACES                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │   MCP    │  │   CLI    │  │   HTTP   │  Entry pts  │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  APPLICATION                            │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │  Commands    │  │   Queries    │  Use Cases         │
│  │ (Write)      │  │  (Read)      │  (CQRS)            │
│  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    DOMAIN                               │
│  ┌────────┐ ┌────────────┐ ┌──────────┐ ┌───────────┐  │
│  │Entities│ │Value Objs  │ │Services  │ │Protocols  │  │
│  │        │ │            │ │(Search,  │ │(Ports)    │  │
│  │        │ │            │ │ Summary) │ │           │  │
│  └────────┘ └────────────┘ └──────────┘ └───────────┘  │
│         NO EXTERNAL DEPENDENCIES                        │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────┐
│               INFRASTRUCTURE                            │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│  │Persistence │ │    LLM     │ │   Config   │ Adapters │
│  │ (SQLite)   │ │(llama.cpp) │ │(Settings)  │          │
│  └────────────┘ └────────────┘ └────────────┘          │
└─────────────────────────────────────────────────────────┘
```

## Dependency Rule
```
Interfaces → Application → Domain ← Infrastructure
```

- **Domain** has ZERO external dependencies (pure Python)
- **Application** depends only on Domain (protocols, entities)
- **Infrastructure** implements Domain protocols
- **Interfaces** wire everything together

## Key Principles

### 1. Domain Protocols (Ports)
```python
# Domain defines what it NEEDS
class EmbeddingProtocol(Protocol):
    async def embed(self, text: str) -> Result[EmbeddingVector, EmbeddingError]: ...
```

### 2. Infrastructure Implements Protocols (Adapters)
```python
# Infrastructure provides HOW
class LlamaCppEmbeddingService:
    async def embed(self, text: str) -> Result[EmbeddingVector, EmbeddingError]:
        # Uses llama-cpp-python
```

### 3. Application Uses Protocols
```python
# Application depends on abstraction
class SearchSessionsQuery:
    def __init__(self, search_service: SearchProtocol):  # Protocol, not concrete
        ...
```

### 4. Composition Root Wires Everything
```python
# interfaces/mcp/server.py
embedding_service = LlamaCppEmbeddingService()
search_service = LlamaCppSearchService(embedding_service, search_repo, summary_service)
search_query = SearchSessionsQuery(search_service)
```

## Benefits

### Testability
- Domain logic testable without any mocks
- Application layer testable with protocol mocks
- Infrastructure testable with real dependencies (integration tests)
- Easy to swap implementations for testing

### Flexibility
- Swap SQLite → PostgreSQL by implementing `SessionRepository`
- Swap llama.cpp → transformers by implementing `EmbeddingProtocol`
- Add new interfaces (HTTP, gRPC) without touching domain

### Maintainability
- Clear separation of concerns
- Business logic isolated from technical details
- Easy to understand and modify

## Consequences

### Positive
- Highly testable (unit tests for domain/application, integration for infrastructure)
- Swappable implementations
- Clear boundaries between layers
- Follows SOLID (especially DIP)

### Negative
- More upfront structure (more files, indirection)
- Requires discipline to maintain boundaries
- Composition root can become complex

## Implementation Guidelines

### DO
- Define protocols in `domain/repositories/`, `domain/services/`
- Keep domain pure (no imports from infrastructure)
- Use dependency injection via constructor
- Write unit tests against protocols

### DON'T
- Import infrastructure in domain layer
- Put business logic in infrastructure
- Skip protocols for "simple" cases
- Use concrete types in application layer

## Consequences for Testing

| Layer | Test Type | Dependencies |
|-------|-----------|--------------|
| Domain | Unit | None (pure) |
| Application | Unit | Mock protocols |
| Infrastructure | Integration | Real DB, real model |
| Interfaces | Contract | Running server |

## References
- [Ports and Adapters](https://alistair.cockburn.us/hexagonal-architecture/)
- [Dependency Inversion Principle](https://en.wikipedia.org/wiki/Dependency_inversion_principle)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)