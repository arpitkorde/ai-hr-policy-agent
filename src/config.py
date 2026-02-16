"""Application configuration using Pydantic Settings."""

from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Explicitly load .env into os.environ BEFORE Pydantic reads them
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE, override=False)


class Settings(BaseSettings):
    """Centralized configuration loaded from environment variables."""

    # --- Google Gemini ---
    google_api_key: str = Field(..., description="Google API key for Gemini")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model name")

    # --- Embeddings & Reranking ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model for embeddings",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace cross-encoder model for reranking",
    )

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(default="./chroma_db", description="ChromaDB storage path")
    chroma_collection_name: str = Field(default="hr_policies", description="ChromaDB collection")

    # --- RAG ---
    chunk_size: int = Field(default=1000, description="Text chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    top_k_retrieval: int = Field(default=20, description="Number of chunks from vector search")
    top_k_rerank: int = Field(default=5, description="Number of chunks after reranking")

    # --- Server ---
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")

    model_config = {"extra": "ignore"}


# Singleton instance
settings = Settings()
