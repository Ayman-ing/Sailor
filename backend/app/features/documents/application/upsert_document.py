from typing import List

from app.features.documents.domain.entities import DocumentChunk
from app.features.documents.domain.repository_interface import EmbeddingRepository
from app.shared.exceptions import VectorStoreError
from app.core.logger import get_logger

logger = get_logger(__name__)


class UpsertDocument:
    
    def __init__(self, embedding_repo: EmbeddingRepository):
        self.embedding_repo = embedding_repo
    
    async def execute(
        self,
        user_id: str,
        course_id: str,
        chunks: List[DocumentChunk],
        dense_embeddings: List[List[float]],
        sparse_embeddings: List[List[float]]
    ) -> int:
        if not chunks:
            return 0
        
        try:
            indexed_ids = await self.embedding_repo.store_chunks(
                user_id=user_id,
                course_id= course_id,
                chunks=chunks,
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings
            )
            
            return len(indexed_ids)
            
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise VectorStoreError(f"Could not upsert chunks: {str(e)}")
