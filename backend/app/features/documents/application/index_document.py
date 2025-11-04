"""Use case for generating embeddings and indexing document chunks in a vector store."""

from typing import List, Tuple
import markdown
from bs4 import BeautifulSoup

from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.repository_interface import EmbeddingRepository
from app.shared.exceptions import EmbeddingError, VectorStoreError
from app.core.logger import get_logger
from app.core.config import settings
from app.core.model_manager import get_model_manager
from qdrant_client.http.models import SparseVector

logger = get_logger(__name__)


class IndexDocument:
    """
    Generates dense and sparse embeddings for document chunks and stores them.
    Uses preloaded models from the model manager.
    """
    
    def __init__(self, embedding_repo: EmbeddingRepository):
        self.embedding_repo = embedding_repo
        self.dense_model_name = settings.embedding_model
        # Get the preloaded models from the model manager
        self.model_manager = get_model_manager()
        self.dense_embedder = self.model_manager.dense_embedder
        self.sparse_embedder = self.model_manager.sparse_embedder
    
    async def execute(
        self, 
        user_id: str,
        document_id: str, 
        chunks: List[DocumentChunk]
    ) -> int:
        if not chunks:
            logger.warning(f"No chunks provided for indexing document {document_id}. Skipping.")
            return 0
            
        try:
            logger.info(f"Starting indexing for {len(chunks)} chunks from document {document_id}")
            
            # 1. Generate both dense and sparse embeddings
            dense_embeddings, sparse_embeddings = await self._generate_embeddings(chunks)
            
            # 2. Store the chunks and both sets of embeddings
            indexed_ids = await self.embedding_repo.store_chunks(
                user_id=user_id,
                document_id=document_id,
                chunks=chunks,
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings
            )
            
            logger.info(f"Successfully indexed {len(indexed_ids)} chunks in vector store.")
            return len(indexed_ids)
            
        except (EmbeddingError, VectorStoreError) as e:
            logger.error(f"A known error occurred during indexing: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while indexing document {document_id}: {e}", exc_info=True)
            raise VectorStoreError(f"An unexpected error occurred during indexing: {str(e)}")


    async def _generate_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[List[List[float]], List[SparseVector]]:
        """Generates both dense and sparse embeddings for a list of chunks."""
        try:
            logger.info(f"Cleaning text for {len(chunks)} chunks...")
            
            # --- Generate Dense Embeddings ---
            logger.info(f"Generating dense embeddings using model '{self.dense_model_name}'...")
            dense_vectors = self.dense_embedder.embed_batch([chunk.content for chunk in chunks]).tolist()
            
            # --- Generate Sparse Embeddings ---
            logger.info("Generating sparse embeddings using SPLADE model...")
            sparse_vectors = self.sparse_embedder.generate_sparse_vectors([chunk.content for chunk in chunks])

            logger.info("Successfully generated dense and sparse embeddings.")
            return dense_vectors, sparse_vectors
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")
