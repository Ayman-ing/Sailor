"""Use case for generating embeddings and indexing document chunks in a vector store."""

from typing import List

from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.repository_interface import EmbeddingRepository
from app.shared.exceptions import EmbeddingError, VectorStoreError
from app.core.logger import get_logger
from app.core.config import settings
from chonkie import SentenceTransformerEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

logger = get_logger(__name__)


class IndexDocument:
    """
    Generates dense and sparse embeddings for document chunks and stores them in the vector database
    via the EmbeddingRepository.
    """
    
    def __init__(self, embedding_repo: EmbeddingRepository):
        """
        Initializes the use case with a repository for storing embeddings.
        
        Args:
            embedding_repo: An implementation of the EmbeddingRepository interface.
        """
        self.embedding_repo = embedding_repo
        self.dense_model_name = settings.embedding_model
        self.sparse_model_name = "prithvida/Splade_PP_en_v1"  # FastEmbed sparse model
    
    async def execute(
        self, 
        user_id: str,
        document_id: str, 
        chunks: List[DocumentChunk]
    ) -> int:
        """
        Generates dense and sparse embeddings for a list of chunks and stores them.
        
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
            logger.info(f"Starting hybrid indexing for {len(chunks)} chunks from document {document_id}")
            
            # 1. Generate dense embeddings
            dense_embeddings = await self._generate_dense_embeddings(chunks)
            
            # 2. Generate sparse embeddings
            sparse_embeddings = await self._generate_sparse_embeddings(chunks)
            
            # 3. Store the chunks with both dense and sparse embeddings
            indexed_ids = await self.embedding_repo.store_chunks(
                user_id=user_id,
                document_id=document_id,
                chunks=chunks,
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings
            )
            
            logger.info(f"Successfully indexed {len(indexed_ids)} chunks with hybrid embeddings.")
            return len(indexed_ids)
            
        except (EmbeddingError, VectorStoreError) as e:
            # These are expected errors, re-raise them
            logger.error(f"A known error occurred during indexing: {e}")
            raise
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred while indexing document {document_id}: {e}")
            raise VectorStoreError(f"An unexpected error occurred during indexing: {str(e)}")

    async def _generate_dense_embeddings(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        """Generates dense embeddings using Chonkie's SentenceTransformer wrapper."""
        try:
            logger.info(f"Generating dense embeddings for {len(chunks)} chunks using '{self.dense_model_name}'...")
            
            # Initialize Chonkie's embedding class
            embedder = SentenceTransformerEmbeddings(self.dense_model_name)
            
            # Get the text content from each chunk
            texts_to_embed = [chunk.content for chunk in chunks]
            
            # Generate embeddings in a single batch call
            embedding_vectors = embedder.embed_batch(texts_to_embed)
            
            # Convert to list of lists
            result = embedding_vectors.tolist() if hasattr(embedding_vectors, 'tolist') else list(embedding_vectors)
            logger.info(f"Generated {len(result)} dense embeddings successfully.")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate dense embeddings: {e}")
            raise EmbeddingError(f"Dense embedding generation failed: {str(e)}")

    async def _generate_sparse_embeddings(self, chunks: List[DocumentChunk]) -> List[SparseVector]:
        """Generates sparse embeddings using FastEmbed's SPLADE model."""
        try:
            logger.info(f"Generating sparse embeddings for {len(chunks)} chunks using '{self.sparse_model_name}'...")
            
            # Initialize FastEmbed sparse model
            sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)
            
            # Get the text content from each chunk
            texts_to_embed = [chunk.content for chunk in chunks]
            
            # Generate sparse embeddings
            sparse_embeddings = []
            
            # FastEmbed returns a generator, process in batches
            for embedding in sparse_model.embed(texts_to_embed, batch_size=32):
                # Convert to SparseVector format for Qdrant
                indices = embedding.indices.tolist()
                values = embedding.values.tolist()
                
                sparse_embeddings.append(
                    SparseVector(
                        indices=indices,
                        values=values
                    )
                )
            
            logger.info(f"Generated {len(sparse_embeddings)} sparse embeddings successfully.")
            return sparse_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate sparse embeddings: {e}")
            raise EmbeddingError(f"Sparse embedding generation failed: {str(e)}")