"""BERT Cross-Encoder Reranker.

Uses a pre-trained BERT cross-encoder model (PyTorch) to rerank
retrieved documents for improved relevance. This dramatically
improves RAG accuracy over pure vector similarity.

Architecture:
    Query → ChromaDB (top-k=20) → BERT Reranker (top-k=5) → LLM
"""

import logging

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)


class BERTReranker:
    """Reranks retrieved documents using a BERT cross-encoder model.

    Cross-encoders jointly encode the query and document together,
    producing more accurate relevance scores than bi-encoder similarity.
    Uses PyTorch under the hood via the sentence-transformers library.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (BERT-based, trained on MS MARCO)
    """

    def __init__(self, model_name: str | None = None):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
                        Defaults to config setting.
        """
        model_name = model_name or settings.reranker_model
        logger.info(f"Loading BERT cross-encoder reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("Reranker model loaded successfully.")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank documents by relevance to the query.

        Args:
            query: The user's question.
            documents: List of Document objects from initial retrieval.
            top_k: Number of top documents to return after reranking.

        Returns:
            List of dicts with keys: 'document', 'score', 'rank'.
            Sorted by relevance score (highest first).
        """
        top_k = top_k or settings.top_k_rerank

        if not documents:
            logger.warning("No documents provided for reranking.")
            return []

        # Prepare query-document pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs using the BERT cross-encoder (PyTorch inference)
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Combine documents with scores and sort
        scored_docs = [
            {"document": doc, "score": float(score), "rank": 0}
            for doc, score in zip(documents, scores)
        ]
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # Assign ranks and take top-k
        for i, item in enumerate(scored_docs):
            item["rank"] = i + 1

        reranked = scored_docs[:top_k]

        logger.info(
            f"Reranked {len(documents)} documents → top {len(reranked)} "
            f"(best score: {reranked[0]['score']:.4f})"
        )
        return reranked

    def rerank_to_documents(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """Rerank and return only the Document objects (for chain integration).

        Args:
            query: The user's question.
            documents: List of Document objects from initial retrieval.
            top_k: Number of top documents to return.

        Returns:
            List of reranked Document objects.
        """
        reranked = self.rerank(query, documents, top_k)
        return [item["document"] for item in reranked]
