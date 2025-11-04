"""Repository interfaces for the chat feature."""

from abc import ABC, abstractmethod
from typing import List, Optional

from app.features.chat.domain.entities import RetrievedChunk


class RetrieverRepository(ABC):
    """Interface for retrieving relevant document chunks."""
    
    @abstractmethod
    async def retrieve_similar_chunks(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        hybrid_alpha: float = 0.5,
        expand_context: bool = True,
        score_threshold: float = 0.7
    ) -> List[RetrievedChunk]:
        """
        Retrieve similar chunks using hybrid search.
        
        Args:
            user_id: User identifier
            query: The search query
            top_k: Number of results to return
            document_ids: Optional list of document IDs to filter by
            hybrid_alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
            expand_context: Whether to retrieve neighboring chunks for high-scoring results
            score_threshold: Minimum score to trigger context expansion
            
        Returns:
            List of retrieved chunks with scores
        """
        pass
