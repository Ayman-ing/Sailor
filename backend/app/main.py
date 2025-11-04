from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

from app.core.logger import get_logger
from app.core.model_manager import model_manager
from app.api_router import api_router

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    logger.info("Application startup...")
    
    # Load all ML models in a background thread to avoid blocking the event loop
    logger.info("Starting model loading in background thread...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, model_manager.load_models)
    logger.info("Model loading complete.")
    
    yield
    
    logger.info("Application shutdown...")

app = FastAPI(
    title="Sailor API",
    description="AI-powered student assistant",
    version="0.1.0",
    lifespan=lifespan
)

# Include the single, aggregated API router with a global prefix
app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Health"])
async def read_root():
    """Root endpoint for health checks."""
    return {"status": "ok"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint that includes model loading status."""
    models_loaded = (
        model_manager.dense_embedder is not None and
        model_manager.sparse_embedder is not None and
        model_manager.docling_converter is not None
    )
    
    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "dense_model": model_manager.dense_embedder is not None,
        "sparse_model": model_manager.sparse_embedder is not None,
        "docling_converter": model_manager.docling_converter is not None
    }
