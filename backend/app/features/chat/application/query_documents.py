"""Use case for querying documents and retrieving relevant chunks."""

from typing import List

from app.features.chat.domain.entities import QueryContext, RetrievedChunk
from app.features.chat.domain.repository_interface import RetrieverRepository
from app.core.logger import get_logger

logger = get_logger(__name__)


class QueryDocuments:
    """
    Queries the document knowledge base and retrieves relevant chunks.
    This is the retrieval step in the RAG pipeline.
    """
    
    def __init__(self, retriever: RetrieverRepository):
        """
        Initialize with a retriever repository.
        
        Args:
            retriever: An implementation of the RetrieverRepository interface
        """
        self.retriever = retriever
    
    async def execute(self, context: QueryContext) -> List[RetrievedChunk]:
        """
        Execute the query and retrieve relevant document chunks.
        
        Args:
            context: Query context with user_id, query, and retrieval parameters
            
        Returns:
            List of relevant document chunks with scores
        """
        # Validate context
        context.validate()
        
        logger.info(
            f"Querying documents for user '{context.user_id}': '{context.query[:50]}...' "
            f"(top_k={context.top_k}, hybrid_alpha={context.hybrid_alpha}, "
            f"expand_context={context.expand_context}, score_threshold={context.score_threshold})"
        )
        
        # Retrieve similar chunks
        chunks = await self.retriever.retrieve_similar_chunks(
            user_id=context.user_id,
            query=context.query,
            top_k=context.top_k,
            document_ids=context.document_ids,
            hybrid_alpha=context.hybrid_alpha,
            expand_context=context.expand_context,
            score_threshold=context.score_threshold
        )
        
        if not chunks:
            logger.warning(f"No relevant chunks found for query: '{context.query[:50]}...'")
            return []
        
        # Log retrieval results
        logger.info(f"Retrieved {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3], 1):  # Log top 3
            logger.debug(
                f"  {i}. Score: {chunk.score:.3f} | "
                f"Doc: {chunk.document_id[:8]}... | "
                f"Page: {chunk.page_number} | "
                f"Content preview: {chunk.content[:80]}..."
            )
        
        return chunks
