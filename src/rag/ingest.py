"""Document ingestion module.

Loads PDF, DOCX, and TXT files, splits them into semantic chunks
for embedding and storage in the vector database.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config import settings

logger = logging.getLogger(__name__)

# Supported file extensions and their loaders
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


class DocumentIngestor:
    """Handles document loading and chunking for the RAG pipeline.

    Supports PDF, DOCX, and TXT files. Splits documents into overlapping
    chunks using RecursiveCharacterTextSplitter for optimal retrieval.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_document(self, file_path: str | Path) -> list[Document]:
        """Load a single document from file path.

        Args:
            file_path: Path to the document file.

        Returns:
            List of Document objects with page content and metadata.

        Raises:
            ValueError: If file extension is not supported.
            FileNotFoundError: If file does not exist.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in LOADER_MAP:
            supported = ", ".join(LOADER_MAP.keys())
            raise ValueError(
                f"Unsupported file type: {ext}. Supported types: {supported}"
            )

        loader_cls = LOADER_MAP[ext]
        loader = loader_cls(str(path))

        logger.info(f"Loading document: {path.name}")
        documents = loader.load()

        # Enrich metadata with source filename
        for doc in documents:
            doc.metadata["source"] = path.name
            doc.metadata["file_type"] = ext

        logger.info(f"Loaded {len(documents)} pages from {path.name}")
        return documents

    def load_directory(self, dir_path: str | Path) -> list[Document]:
        """Load all supported documents from a directory.

        Args:
            dir_path: Path to the directory containing documents.

        Returns:
            List of Document objects from all files.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        all_documents = []
        for ext in LOADER_MAP:
            for file_path in dir_path.glob(f"*{ext}"):
                try:
                    docs = self.load_document(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to load {file_path.name}: {e}")

        logger.info(f"Loaded {len(all_documents)} total pages from {dir_path}")
        return all_documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks for embedding.

        Args:
            documents: List of loaded Document objects.

        Returns:
            List of chunked Document objects with preserved metadata.
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info(
            f"Split {len(documents)} documents into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def ingest(self, file_path: str | Path) -> list[Document]:
        """Full ingestion pipeline: load + split a single document.

        Args:
            file_path: Path to the document.

        Returns:
            List of chunked Document objects ready for embedding.
        """
        documents = self.load_document(file_path)
        return self.split_documents(documents)

    def ingest_directory(self, dir_path: str | Path) -> list[Document]:
        """Full ingestion pipeline: load + split all documents in a directory.

        Args:
            dir_path: Path to directory containing documents.

        Returns:
            List of chunked Document objects ready for embedding.
        """
        documents = self.load_directory(dir_path)
        return self.split_documents(documents)
