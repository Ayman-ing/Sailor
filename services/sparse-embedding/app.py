"""Sparse Embedding Service - Provides SPLADE sparse embeddings via HTTP."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union
import uvicorn
from fastembed import SparseTextEmbedding
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "prithvida/Splade_PP_en_v1"
MAX_BATCH_SIZE = 32  # Sparse embeddings are very memory-intensive
INTERNAL_BATCH_SIZE = 8  # Process only 8 texts at a time to prevent OOM
model = None
executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global model, executor
    
    logger.info(f"Loading sparse embedding model: {MODEL_NAME}")
    model = SparseTextEmbedding(model_name=MODEL_NAME)
    executor = ThreadPoolExecutor(max_workers=4)
    logger.info("Sparse embedding model loaded")
    
    yield
    
    if executor:
        executor.shutdown(wait=True)
    logger.info("Sparse embedding service shutdown complete")


app = FastAPI(
    title="Sparse Embedding Service",
    description="Provides sparse vector embeddings using FastEmbed",
    version="1.0.0",
    lifespan=lifespan
)


class SparseVector(BaseModel):
    """Sparse vector representation."""
    indices: List[int]
    values: List[float]


class IndexedText(BaseModel):
    """Text with index for order preservation."""
    index: int
    text: str


class EmbedRequest(BaseModel):
    """Request model for embedding generation."""
    texts: Union[List[str], List[IndexedText]] = Field(..., max_length=MAX_BATCH_SIZE)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    {"index": 0, "text": "Hello world"},
                    {"index": 1, "text": "How are you?"}
                ]
            }
        }


class IndexedSparseEmbedding(BaseModel):
    """Sparse embedding with index for order preservation."""
    index: int
    indices: List[int]
    values: List[float]


class EmbedResponse(BaseModel):
    """Response model containing sparse embeddings."""
    embeddings: List[IndexedSparseEmbedding]
    model: str


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate sparse embeddings for the provided texts with internal batching to prevent OOM."""
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
        indexed_texts = [(item.index, item.text) for item in request.texts]
        texts_only = [text for _, text in indexed_texts]
        
        loop = asyncio.get_event_loop()
        sparse_vectors = []
        
        num_batches = (len(texts_only) + INTERNAL_BATCH_SIZE - 1) // INTERNAL_BATCH_SIZE
        
        for i in range(0, len(texts_only), INTERNAL_BATCH_SIZE):
            batch = texts_only[i:i + INTERNAL_BATCH_SIZE]
            batch_num = i // INTERNAL_BATCH_SIZE + 1
            
            logger.debug(f"Processing sparse embedding batch {batch_num}/{num_batches} ({len(batch)} texts)")
            
            batch_embeddings = await loop.run_in_executor(
                executor,
                lambda b=batch: list(model.embed(b))
            )
            
            # Always return indexed format - get corresponding indices for this batch
            batch_indices = [idx for idx, _ in indexed_texts[i:i + INTERNAL_BATCH_SIZE]]
            for idx, embedding in zip(batch_indices, batch_embeddings):
                sparse_vectors.append(IndexedSparseEmbedding(
                    index=idx,
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist()
                ))
        
        logger.info(f"Generated {len(sparse_vectors)} sparse embeddings")
        return EmbedResponse(
            embeddings=sparse_vectors,
            model=MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "loading",
        "model": MODEL_NAME
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Sparse Embedding Service",
        "model": MODEL_NAME,
        "endpoints": {
            "/embed": "POST - Generate sparse embeddings",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )