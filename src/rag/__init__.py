"""RAG pipeline components for HR Policy Agent."""

# Lazy imports — use `from src.rag.ingest import DocumentIngestor` directly
# Eager re-exports removed to avoid Python 3.14 + Pydantic v1 compatibility issues

__all__ = ["DocumentIngestor", "VectorStoreManager", "BERTReranker", "HRPolicyChain"]
