import asyncio
from typing import List, Tuple

from app.features.documents.domain.entities import DocumentChunk
from app.shared.exceptions import EmbeddingError
from app.core.logger import get_logger
from app.core.embedding_client import embedding_client
from app.core.config import settings
from qdrant_client.models import SparseVector

logger = get_logger(__name__)


class EmbedDocument:
    
    def __init__(self):
        self.embedding_client = embedding_client
    
    async def execute(self, chunks: List[DocumentChunk]) -> Tuple[List[List[float]], List[SparseVector]]:
        if not chunks:
            return ([], [])
        
        try:
            dense_task = self._generate_dense_embeddings(chunks)
            sparse_task = self._generate_sparse_embeddings(chunks)
            
            dense_embeddings, sparse_embeddings = await asyncio.gather(dense_task, sparse_task)
            
            return (dense_embeddings, sparse_embeddings)
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise EmbeddingError(f"Embedding failed: {str(e)}")

    async def _generate_dense_embeddings(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        try:
            texts_to_embed = [chunk.content for chunk in chunks]
            BATCH_SIZE = settings.embedding_request_batch_size
            all_embeddings = []
            
            for i in range(0, len(texts_to_embed), BATCH_SIZE):
                batch = texts_to_embed[i:i + BATCH_SIZE]
                batch_embeddings = await self.embedding_client.get_dense_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Dense embedding failed: {str(e)}")

    async def _generate_sparse_embeddings(self, chunks: List[DocumentChunk]) -> List[SparseVector]:
        try:
            texts_to_embed = [chunk.content for chunk in chunks]
            BATCH_SIZE = settings.embedding_request_batch_size
            all_sparse_vectors = []
            
            for i in range(0, len(texts_to_embed), BATCH_SIZE):
                batch = texts_to_embed[i:i + BATCH_SIZE]
                batch_sparse_vectors = await self.embedding_client.get_sparse_embeddings(batch)
                all_sparse_vectors.extend(batch_sparse_vectors)
            
            return all_sparse_vectors
            
        except Exception as e:
            raise EmbeddingError(f"Sparse embedding failed: {str(e)}")
