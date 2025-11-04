"""Singleton manager for embedding models to avoid reloading."""

from typing import Optional
from chonkie import SentenceTransformerEmbeddings
from fastembed import SparseTextEmbedding
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModelsManager:
    """
    Singleton manager for embedding models.
    Ensures models are loaded once and reused across the application.
    """
    
    _instance: Optional['EmbeddingModelsManager'] = None
    _dense_model: Optional[SentenceTransformerEmbeddings] = None
    _sparse_model: Optional[SparseTextEmbedding] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def dense_model(self) -> SentenceTransformerEmbeddings:
        """Get or initialize the dense embedding model."""
        if self._dense_model is None:
            logger.info(f"Loading dense embedding model: {settings.embedding_model}")
            self._dense_model = SentenceTransformerEmbeddings(settings.embedding_model)
            logger.info("Dense embedding model loaded successfully")
        return self._dense_model
    
    @property
    def sparse_model(self) -> SparseTextEmbedding:
        """Get or initialize the sparse embedding model."""
        if self._sparse_model is None:
            logger.info(f"Loading sparse embedding model: {settings.sparse_embedding_model}")
            self._sparse_model = SparseTextEmbedding(model_name=settings.sparse_embedding_model)
            logger.info("Sparse embedding model loaded successfully")
        return self._sparse_model
    
    def reload_models(self):
        """Force reload of all models (useful for testing or model updates)."""
        logger.info("Reloading all embedding models")
        self._dense_model = None
        self._sparse_model = None


# Global instance
embedding_models_manager = EmbeddingModelsManager()
