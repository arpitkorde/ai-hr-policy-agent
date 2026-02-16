"""HR Policy RAG Chain.

Orchestrates the full RAG pipeline:
Query → Vector Search → BERT Reranking → Gemini Generation → Cited Answer

Includes observability logging for production monitoring.
"""

import logging
import time
from dataclasses import dataclass, field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from src.config import settings
from src.rag.vector_store import VectorStoreManager
from src.rag.reranker import BERTReranker
from src.rag.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured result from the RAG chain."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    chunks_retrieved: int = 0
    chunks_after_rerank: int = 0
    prompt_version: str = ""


class HRPolicyChain:
    """End-to-end RAG chain for HR policy question answering.

    Pipeline:
        1. Retrieve top-k chunks from ChromaDB (vector similarity)
        2. Rerank using BERT cross-encoder for precision
        3. Generate answer with Gemini using reranked context
        4. Return structured result with citations and metrics
    """

    def __init__(
        self,
        vector_store: VectorStoreManager | None = None,
        reranker: BERTReranker | None = None,
        prompt_version: str = "v2.0",
    ):
        # Initialize components (lazy loading if not provided)
        self.vector_store = vector_store or VectorStoreManager()
        self.reranker = reranker or BERTReranker()
        self.prompt_version = prompt_version

        # Initialize Gemini LLM
        logger.info(f"Initializing Gemini model: {settings.gemini_model}")
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.3,
            max_output_tokens=2048,
        )

        # Load prompt template
        self.prompt = get_prompt(prompt_version)
        logger.info(f"HR Policy Chain initialized (prompt: {prompt_version})")

    def query(self, question: str) -> QueryResult:
        """Run the full RAG pipeline for a question.

        Args:
            question: The employee's question about HR policies.

        Returns:
            QueryResult with answer, sources, and observability metrics.
        """
        start_time = time.time()

        # Step 1: Retrieve from vector store
        logger.info(f"Step 1: Retrieving chunks for: '{question[:80]}...'")
        retrieved_docs = self.vector_store.similarity_search(
            question, k=settings.top_k_retrieval
        )

        if not retrieved_docs:
            return QueryResult(
                answer="I don't have any policy documents loaded yet. "
                       "Please ask your HR team to upload the relevant policies.",
                query=question,
                latency_ms=self._elapsed_ms(start_time),
            )

        # Step 2: Rerank with BERT cross-encoder
        logger.info(f"Step 2: Reranking {len(retrieved_docs)} chunks")
        reranked_docs = self.reranker.rerank_to_documents(
            question, retrieved_docs, top_k=settings.top_k_rerank
        )

        # Step 3: Build context from reranked documents
        context = self._build_context(reranked_docs)
        sources = self._extract_sources(reranked_docs)

        # Step 4: Generate answer with Gemini
        logger.info("Step 3: Generating answer with Gemini")
        messages = self.prompt.format_messages(
            context=context, question=question
        )
        response = self.llm.invoke(messages)

        # Build result with observability metrics
        latency = self._elapsed_ms(start_time)
        result = QueryResult(
            answer=response.content,
            sources=sources,
            query=question,
            latency_ms=latency,
            tokens_used=response.usage_metadata.get("total_tokens", 0)
            if hasattr(response, "usage_metadata") and response.usage_metadata
            else 0,
            chunks_retrieved=len(retrieved_docs),
            chunks_after_rerank=len(reranked_docs),
            prompt_version=self.prompt_version,
        )

        # Observability log
        logger.info(
            f"Query completed | latency={latency:.0f}ms | "
            f"chunks={result.chunks_retrieved}→{result.chunks_after_rerank} | "
            f"tokens={result.tokens_used} | prompt={self.prompt_version}"
        )

        return result

    def _build_context(self, documents: list[Document]) -> str:
        """Format reranked documents into context string for the LLM."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(
                f"[Document {i}: {source}, Page {page}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, documents: list[Document]) -> list[dict]:
        """Extract source metadata from documents for citations."""
        sources = []
        seen = set()
        for doc in documents:
            source_name = doc.metadata.get("source", "Unknown")
            if source_name not in seen:
                seen.add(source_name)
                sources.append({
                    "document": source_name,
                    "page": doc.metadata.get("page", "N/A"),
                    "file_type": doc.metadata.get("file_type", "unknown"),
                })
        return sources

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.time() - start_time) * 1000
