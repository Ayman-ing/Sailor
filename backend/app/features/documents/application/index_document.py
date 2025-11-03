"""Use case for generating embeddings and indexing document chunks in a vector store."""

from typing import List

from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.repository_interface import EmbeddingRepository
from app.shared.exceptions import EmbeddingError, VectorStoreError
from app.core.logger import get_logger
from app.core.config import settings
from chonkie import SentenceTransformerEmbeddings

logger = get_logger(__name__)


class IndexDocument:
    """
    Generates embeddings for document chunks and stores them in the vector database
    via the EmbeddingRepository.
    """
    
    def __init__(self, embedding_repo: EmbeddingRepository):
        """
        Initializes the use case with a repository for storing embeddings.
        
        Args:
            embedding_repo: An implementation of the EmbeddingRepository interface.
        """
        self.embedding_repo = embedding_repo
        self.embedding_model_name = settings.embedding_model
    
    async def execute(
        self, 
        user_id: str,
        document_id: str, 
        chunks: List[DocumentChunk]
    ) -> int:
        """
        Generates embeddings for a list of chunks and stores them.
        
        Args:
            user_id: The ID of the user who owns the document.
            document_id: The ID of the document the chunks belong to.
            chunks: The list of DocumentChunk objects to be indexed.
            
        Returns:
            The number of chunks that were successfully indexed.
        """
        if not chunks:
            logger.warning(f"No chunks provided for indexing document {document_id}. Skipping.")
            return 0
            
        try:
            logger.info(f"Starting indexing for {len(chunks)} chunks from document {document_id}")
            
            # 1. Generate embeddings for all chunks in a batch
            embeddings = await self._generate_embeddings(chunks)
            
            # 2. Store the chunks and their embeddings in the vector store
            indexed_ids = await self.embedding_repo.store_chunks(
                user_id=user_id,
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings
            )
            
            logger.info(f"Successfully indexed {len(indexed_ids)} chunks in vector store.")
            return len(indexed_ids)
            
        except (EmbeddingError, VectorStoreError) as e:
            # These are expected errors, re-raise them
            logger.error(f"A known error occurred during indexing: {e}")
            raise
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred while indexing document {document_id}: {e}")
            raise VectorStoreError(f"An unexpected error occurred during indexing: {str(e)}")

    async def _generate_embeddings(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """Generates embeddings for a list of chunks using Chonkie's embedding wrapper."""
        try:
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks using model '{self.embedding_model_name}'...")
            
            # Initialize Chonkie's embedding class
            embedder = SentenceTransformerEmbeddings(self.embedding_model_name)
            
            # Get the text content from each chunk
            texts_to_embed = [chunk.content for chunk in chunks]
            
            # Generate embeddings in a single batch call
            embedding_vectors = embedder.embed_batch(texts_to_embed)
            
            # Convert numpy array to a list of lists
            return embedding_vectors.tolist() if hasattr(embedding_vectors, 'tolist') else list(embedding_vectors)
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings with Chonkie: {e}")
            # Wrap the specific error in our domain exception
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")