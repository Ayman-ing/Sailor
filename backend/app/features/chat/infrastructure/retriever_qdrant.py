"""Qdrant implementation of the RetrieverRepository interface."""

from typing import List, Optional

from app.features.chat.domain.entities import RetrievedChunk
from app.features.chat.domain.repository_interface import RetrieverRepository
from app.core.qdrant_client import QdrantManager, generate_user_collection_name
from app.core.embedding_models import embedding_models_manager
from app.core.logger import get_logger
from app.shared.exceptions import VectorStoreError

logger = get_logger(__name__)


class RetrieverQdrant(RetrieverRepository):
    """
    Retrieves relevant document chunks from Qdrant using hybrid search.
    """
    
    def __init__(self, qdrant_manager: QdrantManager):
        self.qdrant = qdrant_manager
    
    async def retrieve_similar_chunks(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        hybrid_alpha: float = 0.7,
        expand_context: bool = True,
        score_threshold: float = 0.7
    ) -> List[RetrievedChunk]:
        """
        Retrieve similar chunks using hybrid search (dense + sparse embeddings).
        For high-scoring chunks, also retrieves neighboring chunks for better context.
        
        Args:
            user_id: User identifier
            query: The search query
            top_k: Number of results to return
            document_ids: Optional list of document IDs to filter by
            hybrid_alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
            expand_context: Whether to retrieve neighboring chunks for high-scoring results
            score_threshold: Minimum score to trigger context expansion (default: 0.7)
            
        Returns:
            List of retrieved chunks with scores, deduplicated and reranked
        """
        try:
            logger.info(f"Retrieving similar chunks for query: '{query[:50]}...' (top_k={top_k})")
            
            # 1. Generate query embeddings
            dense_embedding = await self._generate_dense_embedding(query)
            sparse_embedding = await self._generate_sparse_embedding(query)
            
            # 2. Search in Qdrant
            collection_name = generate_user_collection_name(user_id)
            
            # Check if collection exists
            if not await self.qdrant.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist. Returning empty results.")
                return []
            
            search_results = await self._search_qdrant(
                collection_name=collection_name,
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                top_k=top_k,
                document_ids=document_ids,
                hybrid_alpha=hybrid_alpha
            )
            
            # 3. Convert to domain entities
            retrieved_chunks = self._convert_to_chunks(search_results)
            
            # 4. Expand context for high-scoring chunks
            if expand_context and retrieved_chunks:
                retrieved_chunks = await self._expand_context_for_high_scores(
                    collection_name=collection_name,
                    chunks=retrieved_chunks,
                    score_threshold=score_threshold,
                    dense_embedding=dense_embedding,
                    user_id=user_id
                )
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks (after context expansion and deduplication)")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}", exc_info=True)
            raise VectorStoreError(f"Retrieval failed: {str(e)}")
    
    async def _generate_dense_embedding(self, text: str) -> List[float]:
        """Generate dense embedding for the query."""
        try:
            embedder = embedding_models_manager.dense_model
            embedding = embedder.embed(text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.error(f"Failed to generate dense embedding: {e}")
            raise VectorStoreError(f"Dense embedding generation failed: {str(e)}")
    
    async def _generate_sparse_embedding(self, text: str):
        """Generate sparse embedding for the query."""
        try:
            sparse_model = embedding_models_manager.sparse_model
            
            # Generate sparse embedding (returns generator)
            sparse_embeddings = list(sparse_model.embed([text]))
            
            if not sparse_embeddings:
                raise VectorStoreError("No sparse embedding generated")
            
            # Get the first (and only) embedding
            embedding = sparse_embeddings[0]
            
            # Return as a simple object with indices and values
            return {
                'indices': embedding.indices.tolist(),
                'values': embedding.values.tolist()
            }
        except Exception as e:
            logger.error(f"Failed to generate sparse embedding: {e}")
            raise VectorStoreError(f"Sparse embedding generation failed: {str(e)}")
    
    async def _search_qdrant(
        self,
        collection_name: str,
        dense_embedding: List[float],
        sparse_embedding: dict,
        top_k: int,
        document_ids: Optional[List[str]],
        hybrid_alpha: float
    ) -> List[dict]:
        """Perform hybrid search in Qdrant."""
        from qdrant_client import models
        
        client = self.qdrant.get_client()
        
        # Build filter if document_ids specified
        search_filter = None
        if document_ids:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=document_ids),
                    )
                ]
            )
        
        # Perform hybrid search using query API with RRF
        logger.info(f"Performing hybrid search with alpha={hybrid_alpha}")
        
        search_result = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_embedding,
                    using="text-dense",
                    limit=top_k * 2  # Get more candidates for reranking
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_embedding['indices'],
                        values=sparse_embedding['values']
                    ),
                    using="text-sparse",
                    limit=top_k * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),  # Reciprocal Rank Fusion
            limit=top_k,
            query_filter=search_filter,
            with_payload=True
        ).points
        
        return [hit.model_dump() for hit in search_result]
    
    def _convert_to_chunks(self, search_results: List[dict]) -> List[RetrievedChunk]:
        """Convert Qdrant search results to domain entities."""
        chunks = []
        
        for result in search_results:
            payload = result.get('payload', {})
            
            chunk = RetrievedChunk(
                chunk_id=str(result.get('id', '')),
                document_id=payload.get('document_id', ''),
                content=payload.get('content', ''),
                score=float(result.get('score', 0.0)),
                chunk_index=payload.get('chunk_index', 0),
                page_number=payload.get('metadata', {}).get('page_number', 0),
                token_count=payload.get('token_count', 0),
                metadata=payload.get('metadata', {})
            )
            
            chunks.append(chunk)
        
        return chunks
    
    async def _expand_context_for_high_scores(
        self,
        collection_name: str,
        chunks: List[RetrievedChunk],
        score_threshold: float,
        dense_embedding: List[float],
        user_id: str
    ) -> List[RetrievedChunk]:
        """
        For high-scoring chunks, retrieve neighboring chunks (previous and next).
        Deduplicates and reranks the combined results.
        
        Args:
            collection_name: Qdrant collection name
            chunks: Initially retrieved chunks
            score_threshold: Minimum score to trigger expansion
            dense_embedding: Query embedding for reranking
            user_id: User identifier for reranking
            
        Returns:
            Chunks with expanded content for high-scoring results
        """
        client = self.qdrant.get_client()
        
        # Find high-scoring chunks that need context expansion
        high_scoring_chunks = [c for c in chunks if c.score >= score_threshold]
        
        if not high_scoring_chunks:
            logger.info("No high-scoring chunks found. Skipping context expansion.")
            return chunks
        
        logger.info(f"Expanding context for {len(high_scoring_chunks)} high-scoring chunks (score >= {score_threshold})")
        
        # Expand content for each high-scoring chunk
        for chunk in high_scoring_chunks:
            neighbors = await self._get_neighboring_chunks(
                client=client,
                collection_name=collection_name,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index
            )
            
            # Append neighboring chunks' content to the original chunk
            if neighbors:
                original_content = chunk.content
                neighbor_contents = [n.content for n in neighbors]
                chunk.content = original_content + "\n\n" + "\n\n".join(neighbor_contents)
                
                logger.debug(
                    f"Expanded chunk {chunk.chunk_index} with {len(neighbors)} neighbors "
                    f"(document: {chunk.document_id[:8]}...)"
                )
        
        logger.info(f"Context expansion complete for {len(high_scoring_chunks)} chunks")
        
        return chunks
    
    async def _get_neighboring_chunks(
        self,
        client: QdrantManager,
        collection_name: str,
        document_id: str,
        chunk_index: int
    ) -> List[RetrievedChunk]:
        """
        Retrieve the next three chunks based on chunk_index.
        
        Args:
            client: Qdrant client
            collection_name: Collection name
            document_id: Document ID
            chunk_index: Current chunk index
            
        Returns:
            List of next three chunks
        """
        from qdrant_client import models
        
        neighbors = []
        
        # Get next chunk (chunk_index + 1)
        next_1_results = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    ),
                    models.FieldCondition(
                        key="chunk_index",
                        match=models.MatchValue(value=chunk_index + 1)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        
        if next_1_results[0]:
            neighbors.extend(self._convert_to_chunks([p.model_dump() for p in next_1_results[0]]))
        
        # Get second next chunk (chunk_index + 2)
        next_2_results = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    ),
                    models.FieldCondition(
                        key="chunk_index",
                        match=models.MatchValue(value=chunk_index + 2)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        
        if next_2_results[0]:
            neighbors.extend(self._convert_to_chunks([p.model_dump() for p in next_2_results[0]]))
        
        # Get third next chunk (chunk_index + 3)
        next_3_results = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    ),
                    models.FieldCondition(
                        key="chunk_index",
                        match=models.MatchValue(value=chunk_index + 3)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        
        if next_3_results[0]:
            neighbors.extend(self._convert_to_chunks([p.model_dump() for p in next_3_results[0]]))
        
        return neighbors
