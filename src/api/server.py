"""FastAPI server for the HR Policy Agent.

Endpoints:
    POST /upload     — Upload and ingest HR policy documents
    POST /query      — Ask questions about HR policies
    GET  /health     — Health check
    GET  /stats      — Knowledge base statistics
    GET  /prompts    — List available prompt versions
"""

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag.ingest import DocumentIngestor
from src.rag.vector_store import VectorStoreManager
from src.rag.reranker import BERTReranker
from src.rag.chain import HRPolicyChain
from src.rag.prompts import list_prompt_versions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Global instances (initialized on startup) ---
ingestor: DocumentIngestor | None = None
vector_store: VectorStoreManager | None = None
chain: HRPolicyChain | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components on startup."""
    global ingestor, vector_store, chain

    logger.info("Initializing HR Policy Agent components...")
    ingestor = DocumentIngestor()
    vector_store = VectorStoreManager()
    reranker = BERTReranker()
    chain = HRPolicyChain(
        vector_store=vector_store,
        reranker=reranker,
    )
    logger.info("All components initialized successfully.")
    yield
    logger.info("Shutting down HR Policy Agent.")


app = FastAPI(
    title="AI HR Policy Agent",
    description="RAG-powered agent for answering employee questions about HR policies",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    """Request model for policy queries."""
    question: str = Field(..., min_length=3, max_length=1000, description="Employee question")
    prompt_version: str = Field(default="v2.0", description="Prompt template version")


class QueryResponse(BaseModel):
    """Response model for policy queries."""
    answer: str
    sources: list[dict]
    metrics: dict


class UploadResponse(BaseModel):
    """Response model for document uploads."""
    filename: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    version: str


class StatsResponse(BaseModel):
    """Response model for knowledge base statistics."""
    collection_name: str
    document_count: int


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Cloud Run."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest an HR policy document.

    Supports: PDF, DOCX, TXT files.
    The document is chunked, embedded, and stored in the vector database.
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Ingest: load → chunk → embed → store
        chunks = ingestor.ingest(tmp_path)
        vector_store.add_documents(chunks)

        logger.info(f"Ingested '{file.filename}': {len(chunks)} chunks created")

        return UploadResponse(
            filename=file.filename,
            chunks_created=len(chunks),
            message=f"Successfully ingested '{file.filename}' ({len(chunks)} chunks)",
        )
    except Exception as e:
        logger.error(f"Failed to ingest '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query_policies(request: QueryRequest):
    """Ask a question about HR policies.

    The question is processed through the RAG pipeline:
    Vector Search → BERT Reranking → Gemini Generation.
    """
    try:
        result = chain.query(request.question)

        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            metrics={
                "latency_ms": round(result.latency_ms, 1),
                "tokens_used": result.tokens_used,
                "chunks_retrieved": result.chunks_retrieved,
                "chunks_after_rerank": result.chunks_after_rerank,
                "prompt_version": result.prompt_version,
            },
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get knowledge base statistics."""
    stats = vector_store.get_collection_stats()
    return StatsResponse(**stats)


@app.get("/prompts")
async def get_prompts():
    """List available prompt template versions."""
    return {"versions": list_prompt_versions()}
