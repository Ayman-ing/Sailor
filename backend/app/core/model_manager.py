# app/core/model_manager.py

import os
import torch
from app.features.documents.infrastructure.sparse_embedding_splade import SpladeSparseEmbeddingService
from chonkie import SentenceTransformerEmbeddings
from app.core.config import settings
from app.core.logger import get_logger
from docling.document_converter import DocumentConverter

logger = get_logger(__name__)

class ModelManager:
    """
    A manager to handle loading and holding ML models and converters.
    """
    def __init__(self):
        self.dense_embedder = None
        self.sparse_embedder = None
        self.docling_converter = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    
    def load_models(self):
        """
        Loads all required machine learning models and converters.
        This is a synchronous, blocking operation.
        """
        logger.info("Loading machine learning models and converters...")
        
        # Load the dense embedding model on GPU
        logger.info(f"Loading dense model: {settings.embedding_model} on {self.device}")
        self.dense_embedder = SentenceTransformerEmbeddings(
            settings.embedding_model,
            device=self.device
        )
        logger.info("Dense model loaded successfully.")

        # Load the sparse embedding model on GPU with batching
        logger.info(f"Loading sparse (SPLADE) model on {self.device}...")
        self.sparse_embedder = SpladeSparseEmbeddingService(
            model_name="prithivida/Splade_PP_en_v1",
            device=self.device
        )
        logger.info("Sparse (SPLADE) model loaded successfully.")
        
        # Load the docling converter
        logger.info("Loading Docling converter with OCR models...")
        self.docling_converter = DocumentConverter()
        logger.info("Docling converter instance created.")
        
        logger.info("All machine learning models and converters loaded.")

# Create a single global instance of the model manager
model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    return model_manager