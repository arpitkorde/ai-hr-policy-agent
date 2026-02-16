"""ChromaDB vector store management.

Handles embedding storage, retrieval, and collection management
for the HR policy knowledge base.
"""

import logging

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the ChromaDB vector store for HR policy documents.

    Uses HuggingFace sentence-transformers for embedding generation
    and ChromaDB for persistent vector storage and similarity search.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ):
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        # Initialize HuggingFace embeddings (runs locally, no API calls)
        model_name = embedding_model or settings.embedding_model
        logger.info(f"Initializing embeddings with model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize ChromaDB
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info(
            f"ChromaDB initialized: collection='{self.collection_name}', "
            f"persist_dir='{self.persist_directory}'"
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add document chunks to the vector store.

        Args:
            documents: List of chunked Document objects to embed and store.

        Returns:
            List of document IDs assigned by ChromaDB.
        """
        if not documents:
            logger.warning("No documents provided to add.")
            return []

        ids = self.vectorstore.add_documents(documents)
        logger.info(f"Added {len(ids)} chunks to vector store.")
        return ids

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
    ) -> list[Document]:
        """Search for documents similar to the query.

        Args:
            query: The search query text.
            k: Number of results to return (defaults to top_k_retrieval setting).

        Returns:
            List of Document objects ranked by similarity.
        """
        k = k or settings.top_k_retrieval
        results = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        return results

    def similarity_search_with_scores(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Search with relevance scores for reranking.

        Args:
            query: The search query text.
            k: Number of results to return.

        Returns:
            List of (Document, score) tuples ranked by similarity.
        """
        k = k or settings.top_k_retrieval
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        logger.info(
            f"Retrieved {len(results)} scored chunks for query: '{query[:50]}...'"
        )
        return results

    def get_retriever(self, k: int | None = None):
        """Get a LangChain retriever interface for the vector store.

        Args:
            k: Number of documents to retrieve.

        Returns:
            A LangChain retriever object.
        """
        k = k or settings.top_k_retrieval
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_collection_stats(self) -> dict:
        """Get statistics about the current collection.

        Returns:
            Dictionary with collection name and document count.
        """
        collection = self.vectorstore._collection
        return {
            "collection_name": self.collection_name,
            "document_count": collection.count(),
        }

    def delete_collection(self) -> None:
        """Delete the entire collection. Use with caution."""
        self.vectorstore.delete_collection()
        logger.warning(f"Deleted collection: {self.collection_name}")
