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

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.slack.bot import app_handler, init_slack_bot, start_socket_mode
from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
from botbuilder.schema import Activity
from src.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Global instances (initialized on startup) ---
ingestor = None
vector_store = None
chain = None
# Teams Bot components
adapter = None
teams_bot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components on startup."""
    global ingestor, vector_store, chain

    logger.info("Initializing HR Policy Agent components...")

    # Lazy imports to avoid Python 3.14 + Pydantic v1 conflicts at module load
    from src.rag.ingest import DocumentIngestor
    from src.rag.vector_store import VectorStoreManager
    from src.rag.reranker import BERTReranker
    from src.rag.chain import HRPolicyChain
    from src.teams.bot import TeamsBot

    # Initialize RAG components

    ingestor = DocumentIngestor()
    vector_store = VectorStoreManager()
    reranker = BERTReranker()
    chain = HRPolicyChain(
        vector_store=vector_store,
        reranker=reranker,
    )
    init_slack_bot(chain)

    # Initialize Teams Bot
    global adapter, teams_bot

    should_disable_teams_auth = settings.teams_disable_auth
    
    if settings.microsoft_app_id and settings.microsoft_app_password:
        if should_disable_teams_auth:
             logger.warning("TEAMS_DISABLE_AUTH is True. Disabling Teams Authentication for local dev.")
             adapter_settings = BotFrameworkAdapterSettings(app_id="", app_password="")
        else:
             adapter_settings = BotFrameworkAdapterSettings(
                app_id=settings.microsoft_app_id,
                app_password=settings.microsoft_app_password,
            )

        adapter = BotFrameworkAdapter(adapter_settings)
        teams_bot = TeamsBot(chain)
        logger.info(f"Microsoft Teams Bot initialized (Auth Disabled: {should_disable_teams_auth}).")
    else:
        logger.warning("Microsoft Teams credentials not found. Teams Bot disabled.")
    
    # Start Slack Socket Mode in background
    import asyncio
    asyncio.create_task(start_socket_mode())

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
    from src.rag.prompts import list_prompt_versions
    return {"versions": list_prompt_versions()}


@app.post("/slack/events")
async def slack_events(req: Request):
    """Handle Slack Events API requests."""
    if not app_handler:
        raise HTTPException(status_code=503, detail="Slack Bot not configured.")
    return await app_handler.handle(req)


@app.post("/api/messages")
async def messages(req: Request):
    """Handle Microsoft Teams Bot Framework messages."""
    logger.info("Received request to /api/messages")
    if not adapter:
        logger.error("Teams Bot not configured.")
        raise HTTPException(status_code=503, detail="Teams Bot not configured.")

    if "application/json" not in req.headers.get("Content-Type", ""):
        logger.error(f"Invalid Content-Type: {req.headers.get('Content-Type')}")
        raise HTTPException(status_code=415, detail="Unsupported Media Type")

    try:
        body = await req.json()
        activity = Activity().deserialize(body)
        auth_header = req.headers.get("Authorization", "")
        
        response = await adapter.process_activity(activity, auth_header, teams_bot.on_turn)
        if response:
            return response.body
        return None
    except Exception as e:
        logger.error(f"Error processing Teams activity: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
