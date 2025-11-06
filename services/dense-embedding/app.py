"""Dense Embedding Service - Provides sentence transformer embeddings via HTTP."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import torch for GPU detection, fallback gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_BATCH_SIZE = 128  # Dense embeddings can handle larger batches
INTERNAL_BATCH_SIZE = 32  # Internal batching for model inference
model = None
executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global model, executor
    
    device = "cpu"
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
            device = "cuda"
            logger.info("AMD ROCm GPU detected")
        else:
            logger.warning("No GPU detected, using CPU")
    
    logger.info(f"Loading dense embedding model: {MODEL_NAME}")
    logger.info(f"Using device: {device}")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    executor = ThreadPoolExecutor(max_workers=4)
    
    logger.info(f"Dense embedding model loaded (dimension: {model.get_sentence_embedding_dimension()})")
    
    yield
    
    if executor:
        executor.shutdown(wait=True)
    logger.info("Dense embedding service shutdown complete")


app = FastAPI(
    title="Dense Embedding Service",
    description="Provides dense vector embeddings using sentence-transformers",
    version="1.0.0",
    lifespan=lifespan
)


class EmbedRequest(BaseModel):
    """Request model for embedding generation."""
    texts: List[str] = Field(..., max_length=MAX_BATCH_SIZE)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Hello world", "How are you?"]
            }
        }


class EmbedResponse(BaseModel):
    """Response model containing embeddings."""
    embeddings: List[List[float]]
    model: str
    dimension: int


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate dense embeddings for the provided texts."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size {len(request.texts)} exceeds maximum {MAX_BATCH_SIZE}"
        )
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            executor,
            lambda: model.encode(
                request.texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=INTERNAL_BATCH_SIZE
            )
        )
        
        logger.info(f"Generated {len(embeddings)} dense embeddings")
        
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model=MODEL_NAME,
            dimension=model.get_sentence_embedding_dimension()
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "loading",
        "model": MODEL_NAME,
        "dimension": model.get_sentence_embedding_dimension() if model else None
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Dense Embedding Service",
        "model": MODEL_NAME,
        "dimension": model.get_sentence_embedding_dimension() if model else None,
        "endpoints": {
            "/embed": "POST - Generate embeddings",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
